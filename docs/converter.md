# The converter and its tools

```
ok_tekken3_converter/
  src/            the converter: disc → a folder (or pack) of standard files; src/main.rs: t3_convert, a developer tool
  examples/       convert.rs: the whole conversion as OpenKumite runs it
  viewer/         a browser viewer for the converted characters, stages, pictures, sound and movies
```

The converter runs no outside program: the PS1 sound and movies are decoded by `ok_converter_common::psx` (XA
ADPCM, STR/MDEC) and written as FLAC and Motion-JPEG AVI. It was ported from Python tools and still writes those
tools' exact output, so its code reproduces Python's and numpy's arithmetic where the output depends on it
(`pyjson`, `np`, the `*_as_python` tests). A change to its output changes the online fingerprint: every player
converts again.

OpenKumite runs the whole conversion through `convert_all::convert_resumable`. The binaries below are in
`target/release/` (`cargo build --release`).

## `t3_convert`

```sh
t3_convert "Tekken_3_(USA).cue" assets [--only characters,stages,...] [--jobs N]  # N threads (1: none); any image format
t3_convert --fingerprint assets            # only the manifest's fingerprint, again
t3_convert MODULE ARGS...                  # one module by hand (t3_convert lists them)
```

| `--only` name | Module (`src/`) | Writes |
|---|---|---|
| characters | `export_gltf` | `characters/model_NNN.glb`: skinned glTF, every animation at 60 fps (own and shared moves), the embu's; each animation's extras `channels`: every frame's 49 channels as the PS1 decodes them (0x80038B4C), the skin's extras `slots`: the 18 joint slots' places on their parents (P+0xF74, format 12) |
| stages | `export_stages` | `stages/stage_NN.glb`: arenas, Tekken Force and Ball stages, sky, walls |
| pictures | `export_pictures` | `pictures/`: the 2D art as PNG, index maps and palettes |
| media | `export_media` | `audio/` (WAV sounds, FLAC music), `video/` (Motion-JPEG AVI + FLAC soundtrack) |
| gamedata | `export_gamedata` | `gamedata/*.json`: movesets, commands, hit paths, reactions, rules, roster, stages, music, AI, camera, HUD layout |
| effects | `export_effects` | `pictures/effects/`: hit sparks and effects in the game's palettes (reads pictures and gamedata) |
| theater | `export_theater` | `gamedata/theater.json`, `pictures/theater/` |
| shadows | `export_shadows` | `gamedata/shadows.json` |

`manifest.json` records what was converted, when, from which image, and a fingerprint of everything a fight reads
(online play refuses computers whose fingerprints differ).

Shared modules: `disc` (ISO 9660, the BNS archive), `tim` (PS1 pictures), `t3m` / `skel` / `skin` / `anim`
(models and motions), `gamedata_*` (the EXE's tables), `mips` (runs a routine's code to read the arguments of its
draw calls: the screens' layouts come from the code, not typed in), `extract_models`, `assemble`, `atlas`,
`texrender`, `embu`, `camera_motion`; `pyjson` writes JSON exactly as Python's `json` does, `np` does numpy's float arithmetic in fixed
orders (the SVD and 4x4 inverse in `ok_converter_common::np`). Other steps: `extract_models` (OBJ models), `qa_glb` (checks
exported models for stretched triangles). `reference/` holds integer-exact prototypes of tricky routines (the
camera, attack paths): `t3_convert camera_golden assets/tests/camera_golden.json` writes the camera tests' golden data. Tools that read the
disc on their own take it from `T3_BIN` (a disc image in any format), else `Tekken_3_(USA)_(Track 1).bin` at this
repository's root.

To redo a step use `--only`: a module run by hand gets none of convert_all's extra arguments (`export_gltf` needs
`--with-common --with-embu`) and leaves the fingerprint as it was (`t3_convert --fingerprint assets` after).

**Adding data:** find it in the EXE (the reference notes: [reference/](reference/README.md)), export it in the step that owns it
(usually `gamedata_tables`, as JSON with the RAM address of each table), then read it in OpenKumite
(`kumite_sim::data` or `kumite_assets`). The engine never reads the disc.

## Working on it

- `sh check.sh commit` (the pre-commit hook): `cargo fmt` (120 columns, `rustfmt.toml`), clippy with warnings as
  errors (`unwrap` and `expect` denied, tests too), the tests; `sh check.sh push` (the pre-push hook, and the
  default) adds `cargo deny` (`deny.toml`).
- The tests that compare with the disc skip silently unless pointed at it:
  ```sh
  T3_BIN="$PWD/Tekken_3_(USA)_(Track 1).bin" T3_GAMEDATA=<a fresh conversion>/gamedata cargo test --release
  T3_FINGERPRINT_DIR=assets cargo test --release -- --ignored fingerprint_of
  ```
- A change to what the converter writes changes the online fingerprint: every player converts again. Check a change
  against the previous build's conversion (`cmp`, PNGs by pixels; the fingerprint).
- Floats: every platform (macOS, Linux, Windows, wasm) must give the same bits, so the fingerprint: only + - * /,
  `sqrt` in fixed orders (`np`, `ok_converter_common::np`), `ok_converter_common::np::fma` (the CPU's FMA, else
  `libm::fma`), and `sin`, `cos`, `acos`, `atan2`, `pow` through the `libm` crate, never `f64::mul_add`,
  `f64::sin` and the like (the platform's libm: Windows x64's static C runtime gets `fma` wrong on a quarter of
  inputs, and `mul_add` calls it where the CPU target has no FMA). ok_hl_cs_converter's fingerprint is a copy of
  `convert_all`'s: change both.

## Viewer

```sh
serve 8765 viewer      # http://127.0.0.1:8765 (serve: ok_converter_common's, cargo run --release --bin serve there)
```

It reads the converted files through links next to it (not in git): in `viewer/`,
`models -> ../assets/characters`, `stages -> ../assets/stages`, `pictures -> ../assets/pictures`. Create them with
`ln -s`. Its sound and movies read the export's own layout: `t3_convert export_media BIN viewer/media`.
