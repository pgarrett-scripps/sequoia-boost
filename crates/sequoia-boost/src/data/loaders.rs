//! Text-format dataset loaders: libsvm and CSV.

use crate::data::dmatrix::DMatrix;
use crate::error::{Result, SequoiaError};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// Load a libsvm / SVMLight file into a sparse [`DMatrix`].
///
/// Each line is `label idx:value idx:value ...` with **0-based** feature
/// indices, matching XGBoost's reader. The label becomes the matrix labels.
pub fn load_libsvm(path: impl AsRef<Path>) -> Result<DMatrix> {
    let file = std::fs::File::open(path)?;
    read_libsvm(BufReader::new(file))
}

/// Parse libsvm-formatted text from any reader.
pub fn read_libsvm<R: Read>(reader: R) -> Result<DMatrix> {
    let reader = BufReader::new(reader);
    let mut indptr = vec![0usize];
    let mut indices: Vec<u32> = Vec::new();
    let mut values: Vec<f32> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();
    let mut max_index = 0u32;

    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let label_tok = it.next().ok_or_else(|| SequoiaError::Parse {
            line: lineno + 1,
            reason: "missing label".into(),
        })?;
        let label: f32 = label_tok.parse().map_err(|_| SequoiaError::Parse {
            line: lineno + 1,
            reason: format!("invalid label `{label_tok}`"),
        })?;
        labels.push(label);

        for tok in it {
            let (idx_s, val_s) = tok.split_once(':').ok_or_else(|| SequoiaError::Parse {
                line: lineno + 1,
                reason: format!("expected idx:value, got `{tok}`"),
            })?;
            let idx: u32 = idx_s.parse().map_err(|_| SequoiaError::Parse {
                line: lineno + 1,
                reason: format!("invalid index `{idx_s}`"),
            })?;
            let val: f32 = val_s.parse().map_err(|_| SequoiaError::Parse {
                line: lineno + 1,
                reason: format!("invalid value `{val_s}`"),
            })?;
            indices.push(idx);
            values.push(val);
            max_index = max_index.max(idx);
        }
        indptr.push(values.len());
    }

    if labels.is_empty() {
        return Err(SequoiaError::EmptyDataset("libsvm: no rows parsed"));
    }
    let n_cols = (max_index as usize) + 1;
    DMatrix::from_csr(indptr, indices, values, n_cols)?.with_labels(&labels)
}

/// Options controlling CSV parsing.
#[derive(Debug, Clone)]
pub struct CsvOptions {
    /// Whether the first line is a header row to skip.
    pub has_header: bool,
    /// Field delimiter.
    pub delimiter: char,
    /// Column index holding the label, if any. Removed from the feature matrix.
    pub label_column: Option<usize>,
    /// Text treated as a missing value (in addition to empty fields).
    pub na_value: Option<String>,
}

impl Default for CsvOptions {
    fn default() -> Self {
        CsvOptions {
            has_header: true,
            delimiter: ',',
            label_column: Some(0),
            na_value: None,
        }
    }
}

/// Load a numeric CSV into a dense [`DMatrix`] (NaN sentinel for missing).
pub fn load_csv(path: impl AsRef<Path>, opts: &CsvOptions) -> Result<DMatrix> {
    let file = std::fs::File::open(path)?;
    read_csv(BufReader::new(file), opts)
}

/// Parse CSV text from any reader.
pub fn read_csv<R: Read>(reader: R, opts: &CsvOptions) -> Result<DMatrix> {
    let reader = BufReader::new(reader);
    let mut rows: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();
    let mut n_cols: Option<usize> = None;

    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if opts.has_header && lineno == 0 {
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(opts.delimiter).collect();
        let mut feats = Vec::with_capacity(fields.len());
        for (c, raw) in fields.iter().enumerate() {
            let field = raw.trim();
            if Some(c) == opts.label_column {
                let label: f32 = field.parse().map_err(|_| SequoiaError::Parse {
                    line: lineno + 1,
                    reason: format!("invalid label `{field}`"),
                })?;
                labels.push(label);
                continue;
            }
            let is_na = field.is_empty() || opts.na_value.as_deref() == Some(field);
            let v = if is_na {
                f32::NAN
            } else {
                field.parse().map_err(|_| SequoiaError::Parse {
                    line: lineno + 1,
                    reason: format!("invalid value `{field}`"),
                })?
            };
            feats.push(v);
        }
        match n_cols {
            None => n_cols = Some(feats.len()),
            Some(expected) if expected != feats.len() => {
                return Err(SequoiaError::Parse {
                    line: lineno + 1,
                    reason: format!("expected {expected} columns, got {}", feats.len()),
                });
            }
            _ => {}
        }
        rows.push(feats);
    }

    let n_cols = n_cols.ok_or(SequoiaError::EmptyDataset("csv: no data rows"))?;
    let n_rows = rows.len();
    let mut flat = Vec::with_capacity(n_rows * n_cols);
    for r in &rows {
        flat.extend_from_slice(r);
    }
    let d = DMatrix::from_dense(&flat, n_rows, n_cols)?;
    if opts.label_column.is_some() && !labels.is_empty() {
        d.with_labels(&labels)
    } else {
        Ok(d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parse_libsvm() {
        let text = "1 0:1.5 2:3.0\n0 1:2.0\n";
        let d = read_libsvm(Cursor::new(text)).unwrap();
        assert_eq!(d.n_rows(), 2);
        assert_eq!(d.n_cols(), 3);
        assert_eq!(d.labels(), Some(&[1.0f32, 0.0][..]));
        assert_eq!(d.get(0, 0), Some(1.5));
        assert_eq!(d.get(0, 1), None);
        assert_eq!(d.get(0, 2), Some(3.0));
        assert_eq!(d.get(1, 1), Some(2.0));
    }

    #[test]
    fn parse_csv_with_header_and_label() {
        let text = "y,f0,f1\n1.0,0.5,0.25\n0.0,,0.75\n";
        let d = read_csv(Cursor::new(text), &CsvOptions::default()).unwrap();
        assert_eq!(d.n_rows(), 2);
        assert_eq!(d.n_cols(), 2);
        assert_eq!(d.labels(), Some(&[1.0f32, 0.0][..]));
        assert_eq!(d.get(0, 0), Some(0.5));
        assert_eq!(d.get(1, 0), None); // empty field -> missing
        assert_eq!(d.get(1, 1), Some(0.75));
    }

    #[test]
    fn csv_column_mismatch_errors() {
        let text = "y,f0,f1\n1.0,0.5,0.25\n0.0,0.75\n";
        assert!(read_csv(Cursor::new(text), &CsvOptions::default()).is_err());
    }
}
