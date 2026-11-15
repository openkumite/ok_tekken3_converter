//! `cargo run --release --example convert -- "Tekken_3_(USA)_(Track 1).bin" OUT`: the whole conversion as an
//! embedding program (OpenKumite) runs it, through `convert_all::convert_resumable` (track 2 is found beside track 1).

use anyhow::{Context, Result};
use ok_converter_common::io::{Ctx, DiscFile, StdDisc, StdFs, print_line};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [bin, out] = &args[..] else { anyhow::bail!("usage: convert BIN OUT") };
    let track2 = StdDisc(bin.replace("Track 1", "Track 2").into());
    let track2 = track2.0.exists().then_some(&track2 as &dyn DiscFile);
    let log = |line: &str| print_line(line);
    let ctx = Ctx { out: &StdFs(out.into()), log: &log, jobs: ok_converter_common::io::jobs_from(&[]) };
    let manifest = ok_tekken3_converter::convert_all::convert_resumable(&ctx, &StdDisc(bin.into()), track2, "now")?;
    println!("{out}: fingerprint {}", manifest["fingerprint"].as_str().context("no fingerprint")?);
    Ok(())
}
