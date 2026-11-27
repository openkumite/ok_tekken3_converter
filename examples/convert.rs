//! `cargo run --release --example convert -- IMAGE OUT`: the whole conversion as an embedding program (OpenKumite)
//! runs it, through `convert_all::convert_resumable`. IMAGE: the disc image in any format (.cue, .bin, .iso, .chd,
//! .ccd, .mds, .ecm, .pbp); its CD-audio track 2 is found with it.

use anyhow::{Context, Result};
use ok_converter_common::disc::Disc;
use ok_converter_common::io::{Ctx, DiscFile, StdFs, print_line};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [image, out] = &args[..] else { anyhow::bail!("usage: convert IMAGE OUT") };
    let disc = Disc::open_path(image.as_ref())?;
    print_line(&disc.summary());
    let log = |line: &str| print_line(line);
    let ctx = Ctx { out: &StdFs(out.into()), log: &log, jobs: ok_converter_common::io::jobs_from(&[]) };
    let track2 = disc.audio(2).map(|t| t as &dyn DiscFile);
    let manifest = ok_tekken3_converter::convert_all::convert_resumable(&ctx, disc.data()?, track2, &[], "now")?;
    println!("{out}: fingerprint {}", manifest["fingerprint"].as_str().context("no fingerprint")?);
    Ok(())
}
