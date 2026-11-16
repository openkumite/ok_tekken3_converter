//! The disc image: its data track's files (ok_converter_common's ISO 9660 reader, with Tekken 3's hint when a file is
//! not there; extract_models's `iso_files` / `read_file`), the image opened in any format (`image`), and Tekken 3's
//! EXE and BNS archive with its record table (disc).

use anyhow::{Context, Result, bail};
use ok_converter_common::disc::{Disc, Track};
use ok_converter_common::io::DiscFile;
use std::ops::{Deref, DerefMut};
use std::path::Path;

pub const SECTOR: u64 = 2352;
pub const HDR: u64 = 24;
/// The BNS record table in SLUS_004.02 (file offset) and its length.
pub const BNS_TABLE_OFF: usize = 0x14C00;
pub const BNS_COUNT: usize = 303;

/// The data track's files.
pub struct Iso(ok_converter_common::disc::Iso);

impl Deref for Iso {
    type Target = ok_converter_common::disc::Iso;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Iso {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Iso {
    pub fn open(disc: &dyn DiscFile) -> Result<Iso> {
        Ok(Iso(ok_converter_common::disc::Iso::open(disc)?))
    }

    pub fn find(&self, name: &str) -> Result<(u32, u32)> {
        self.0.find(name).with_context(|| format!("{name} is not on the disc (is it Tekken 3 USA, SLUS-00402?)"))
    }

    pub fn read(&mut self, name: &str) -> Result<Vec<u8>> {
        let (lba, size) = self.find(name)?;
        self.read_file(lba, size)
    }
}

/// The disc image at `path`, in any format ok_converter_common reads (.cue/.bin, .iso, .chd, .ccd, .mds, .ecm, .pbp;
/// or in an archive: .zip, .7z, .tar, .gz, .xz, .zst, unpacked into a temporary file), with the files beside it.
pub fn image(path: impl AsRef<Path>) -> Result<Disc> {
    Disc::open_path(path.as_ref())
}

/// The image's data track (track 1).
pub fn track1(path: impl AsRef<Path>) -> Result<Track> {
    Ok(image(path)?.data()?.clone())
}

/// Track 1 and the CD-audio track 2 (when the image keeps it).
pub fn tracks(path: impl AsRef<Path>) -> Result<(Track, Option<Track>)> {
    let disc = image(path)?;
    Ok((disc.data()?.clone(), disc.audio(2).cloned()))
}

/// Tekken 3's EXE, BNS archive and its records (`rec[i]` of the export scripts).
pub struct Game {
    pub exe: Vec<u8>,
    pub bns: Vec<u8>,
    pub rec: Vec<Vec<u8>>,
}

impl Game {
    pub fn load(disc: &dyn DiscFile) -> Result<Game> {
        let iso = &mut Iso::open(disc)?;
        let exe = iso.read("TEKKEN3/SLUS_004.02")?;
        let bns = iso.read("TEKKEN3/TEKKEN3.BNS")?;
        let mut rec = Vec::with_capacity(BNS_COUNT);
        for i in 0..BNS_COUNT {
            let at = BNS_TABLE_OFF + 8 * i;
            if at + 8 > exe.len() {
                bail!("the EXE is too short for its BNS table");
            }
            let (s, z) = (u32le(&exe, at) as usize, u32le(&exe, at + 4) as usize);
            rec.push(py_slice(&bns, s * 2048, s * 2048 + z).to_vec());
        }
        Ok(Game { exe, bns, rec })
    }

    /// The EXE's bytes at a RAM address (file offset = address - 0x80010000 + 0x800).
    pub fn at(&self, addr: u32) -> &[u8] {
        py_slice(&self.exe, (addr.wrapping_sub(0x8000F800)) as usize, usize::MAX)
    }
}

/// Python's `b[a:b]`: clamped to the data, empty when reversed.
pub fn py_slice(d: &[u8], a: usize, b: usize) -> &[u8] {
    let a = a.min(d.len());
    let b = b.min(d.len());
    if a >= b { &[] } else { &d[a..b] }
}

/// disc's image: the T3_BIN environment variable (a disc image in any format), or "Tekken_3_(USA)_(Track 1).bin" at
/// the repository's root.
pub fn bin() -> std::path::PathBuf {
    match std::env::var("T3_BIN") {
        Ok(p) if !p.is_empty() => p.into(),
        _ => Path::new(env!("CARGO_MANIFEST_DIR")).join("Tekken_3_(USA)_(Track 1).bin"),
    }
}

pub fn u32le(d: &[u8], o: usize) -> u32 {
    d.get(o..o + 4).map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}
