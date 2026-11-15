# ok_tekken3_converter

Converts your own Tekken 3 disc (PlayStation, USA, SLUS-00402) into the game data OpenKumite plays (OpenKumite: an
open-source 3D fighting game engine, a separate project): skinned glTF characters and stages, PNG pictures, JSON
game data, WAV/FLAC sound and Motion-JPEG AVI movies.

> **Unofficial.** ok_tekken3_converter is not affiliated with, endorsed or sponsored by Bandai Namco Entertainment.
> TEKKEN is a trademark of Bandai Namco Entertainment Inc.; PlayStation is a trademark of Sony Interactive
> Entertainment. This repository contains no game data: you need your own legally obtained copy, which the
> converter reads on your computer. Do not ask for, or share, disc images or converted files.

**Players do not run it.** OpenKumite links this crate and converts in the game: its Game data screen asks for the
disc image the first time. This repository is the library and the developer tools for working on it.

The Tekken 3 pack is the base: the Tekken 2 fighters (ok_tekken2_converter) and the Counter-Strike levels
(ok_hl_cs_converter) are packs that play beside it.

## The library

```rust
use ok_converter_common::io::{Ctx, StdDisc, StdFs};

let ctx = Ctx { out: &StdFs("OUT".into()), log: &|line| println!("{line}"), jobs: 4 };
let manifest = ok_tekken3_converter::convert_all::convert_resumable(&ctx, &StdDisc(bin.into()), track2, now)?;
```

`convert_resumable` converts everything into a `Vfs` (a folder, or the pack file the browser keeps) and writes
`manifest.json`, with the fingerprint online play compares. An interrupted conversion picks up after its last
finished step. `examples/convert.rs` is the whole program (`cargo run --release --example convert -- BIN OUT`).

The disc image is the raw `.bin` (`Tekken_3_(USA)_(Track 1).bin`; a `.chd`: `chdman extractcd` first; a 2048-byte
`.iso` lacks the sound and movies). The CD-audio track 2 is read beside it. The output is about 1.4 GB.

## Repository

| | |
|---|---|
| `src/` | the converter, one module per step (`convert_all` runs them); `src/main.rs`: `t3_convert`, the developer tool |
| `viewer/` | a browser viewer for the converted characters, stages, pictures, sound and movies |
| `docs/converter.md` | the steps, the developer tool, the viewer, how to check a change |
| `docs/reference/` | the reference notes: the original game and its disc formats |

It depends on [ok_converter_common](../ok_converter_common) (disc and file input/output, PlayStation media,
portable float maths).

`sh check.sh` runs the checks (formatting, clippy, tests, cargo deny); the git hooks and CI run it.

## License

MIT ([LICENSE](LICENSE)). Third-party work: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
