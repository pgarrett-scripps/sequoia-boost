//! Train 72 fixed configs and write the bit patterns of predictions (P),
//! leaf ids (L) and SHAP contributions (S) to `<out_dir>/<config>.bin`.
//! Usage: equiv_dump <out_dir> <nthread>
use sequoia_boost::prelude::*;
use sequoia_verify::{dispatch_path, lcg};
use std::io::Write;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let out_dir = &args[1];
    let nthread: usize = args[2].parse().expect("nthread");
    eprintln!("dispatch: {}", dispatch_path());
    std::fs::create_dir_all(out_dir).expect("out dir");
    let (n, p) = (3000usize, 12usize);
    for &(obj, k) in &[
        ("reg:squarederror", 1usize),
        ("binary:logistic", 2),
        ("multi:softprob", 3),
    ] {
        for &missing in &[false, true] {
            for &seed in &[1u64, 7] {
                for &depth in &[3usize, 6, 9] {
                    for &method in &["hist", "exact"] {
                        let mut s = seed * 1000 + depth as u64 + if missing { 99 } else { 0 };
                        let mut x = vec![0f32; n * p];
                        for v in x.iter_mut() {
                            *v = lcg(&mut s) * 4.0 - 2.0;
                            if missing && lcg(&mut s) < 0.15 {
                                *v = f32::NAN;
                            }
                        }
                        let y: Vec<f32> = (0..n)
                            .map(|i| {
                                let r = &x[i * p..(i + 1) * p];
                                let a = |j: usize| if r[j].is_nan() { 0.3 } else { r[j] };
                                let z = a(0) * 1.5 - a(1) * a(2)
                                    + (a(3) * 2.0).sin()
                                    + 0.3 * lcg(&mut s);
                                match k {
                                    1 => z,
                                    2 => (z > 0.2) as u8 as f32,
                                    _ => (z >= -0.5) as u8 as f32 + (z >= 0.8) as u8 as f32,
                                }
                            })
                            .collect();
                        let d = DMatrix::from_dense(&x, n, p)?.with_labels(&y)?;
                        let mut b = TrainingParams::builder()
                            .objective(obj)
                            .max_depth(depth)
                            .eta(0.2)
                            .seed(seed)
                            .nthread(nthread)
                            .subsample(0.8)
                            .colsample_bytree(0.8)
                            .tree_method(if method == "hist" {
                                TreeMethod::Hist
                            } else {
                                TreeMethod::Exact
                            });
                        if k > 2 {
                            b = b.num_class(k);
                        }
                        let m = train(&b.build()?, &d, 20)?;
                        let pr: Vec<u32> = m.predict(&d)?.iter().map(|v| v.to_bits()).collect();
                        let lf = m.predict_leaf(&d)?;
                        let sh: Vec<u32> = m
                            .predict_contribs(&d)?
                            .iter()
                            .map(|v| v.to_bits())
                            .collect();
                        let name = format!(
                            "{out_dir}/{}_{missing}_{seed}_{depth}_{method}.bin",
                            obj.replace(':', "-")
                        );
                        let mut f = std::fs::File::create(&name).expect("create");
                        for (tag, bits) in [("P", pr), ("L", lf), ("S", sh)] {
                            writeln!(f, "{tag} {}", bits.len()).unwrap();
                            for v in bits {
                                f.write_all(&v.to_le_bytes()).unwrap();
                            }
                            writeln!(f).unwrap();
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
