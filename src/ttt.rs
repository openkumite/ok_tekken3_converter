//! Tekken 3's fighters in Tekken Tag Tournament (OpenKumite's crossover, installed into Tekken Tag's extras): Gon and
//! Doctor B. as Tekken Tag fighters, Crow (Tekken Force's soldiers) as one in no select. Their Tekken 3 move system
//! is kept and renumbered into Tekken Tag's aliases; their bodies and motions are turned onto Tekken Tag's joint
//! axes. Reads Tekken 3's game root (this converter's output) and Tekken Tag's (ok_tekken_tag_converter's, through
//! `ctx.out`: the extras over it), writes the files that add them to Tekken Tag into `ctx.out`. Also Tekken Ball's
//! rules, court, sounds and camera ([`ball`]), Tekken Force's stages, soldiers, bosses, pictures, music and sounds
//! ([`force`]), and Tekken 3's HUD those two modes draw ([`hud`]).
//!
//! Tekken 3's aliases and Tekken Tag's are numbered apart: they are paired on the 14 fighters both games have
//! ([`pairing`]): their attacks by (frames, hit mask, active frames) when unique on both sides, the reactions those
//! deal, the reactions' 14 victim slots (both games keep Tekken 3's slot order), then the paired moves' next moves
//! and command lists.

use anyhow::{Context, Result, bail};
use image::{Rgba, RgbaImage};
use ok_converter_common::io::{Ctx, Vfs};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

/// Tekken 3's character ids, and the body and moveset ids they take in Tekken Tag (past its 34 bodies and 32
/// movesets): Gon and Doctor B. (selectable), Crow (Tekken Force's soldiers: in no select).
pub const FIGHTERS: [(usize, usize, usize); 3] = [(17, 34, 32), (19, 35, 33), (21, 36, 34)];
/// The first [`FIGHTERS`] are on the select screen.
const SELECTABLE: usize = 2;
/// The fighters both games have: Tekken 3's characters (and Tekken Tag's bodies) 0-13.
const SHARED: usize = 14;
/// Tekken Tag's alias count: aliases of the crossover fighters' own moves without a pair are numbered from here.
const TTT_ALIASES: usize = 5532;
/// The aliases the engines name, Tekken 3's to Tekken Tag's: the stance, the crouch, the K.O. collapse, the round
/// starts (held punch, kick, later rounds), the win poses (rules.json `win_lose_motions`, rounds.json `win_poses`).
const FIXED: [(usize, usize); 10] = [
    (3, 58),
    (0x28, 80),
    (0xD66, 4769),
    (0, 4),
    (2, 6),
    (1, 5),
    (3431, 4770),
    (3432, 4771),
    (3433, 4772),
    (3434, 4773),
];
/// The height (game units) the select lifts a crossover fighter's stance's top to when it is lower (Gon's, under the
/// select's camera on the floor): his head in the picture, a little under Tekken Tag's own.
const SELECT_TOP: u64 = 1300;
/// The air reactions by meaning, Tekken 3's entry to Tekken Tag's (0x8001A630 / 0x500678): a juggle hit, a spike,
/// the late fall, the side hits, the cancel-driven one; Tekken Tag's own spike and fall variants (7, 5, 6) take the
/// same.
const AIR: [(usize, usize); 8] = [(0, 0), (1, 1), (3, 2), (4, 3), (5, 4), (2, 5), (2, 6), (1, 7)];

fn read(fs: &dyn Vfs, path: &str) -> Result<Value> {
    serde_json::from_slice(&fs.read(path).with_context(|| format!("reading {path}"))?)
        .with_context(|| format!("parsing {path}"))
}

fn write(ctx: &Ctx, path: &str, v: &Value) -> Result<()> {
    ctx.out.write(path, &serde_json::to_vec(v)?)
}

fn n(v: &Value) -> Option<usize> {
    v.as_u64().map(|x| x as usize)
}

fn arr(v: &Value) -> &[Value] {
    v.as_array().map_or(&[], |a| a.as_slice())
}

/// The first word of a roster name ("Eddy / Tiger": "eddy").
fn short(name: &str) -> String {
    name.split(" /").next().unwrap_or(name).trim().to_lowercase()
}

// ---------------------------------------------------------------------------------------------------------------
// The bodies: each joint's axes turned onto Tekken Tag's

/// A quaternion (x, y, z, w), glTF's order.
type Q = [f64; 4];

fn qmul(a: Q, b: Q) -> Q {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

fn conj(q: Q) -> Q {
    [-q[0], -q[1], -q[2], q[3]]
}

/// The rotation matrix of a unit quaternion, rows.
fn qmat(q: Q) -> [[f64; 3]; 3] {
    let [x, y, z, w] = q;
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

fn rotate(q: Q, v: [f64; 3]) -> [f64; 3] {
    let m = qmat(q);
    std::array::from_fn(|r| m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2])
}

/// Each joint's turn D from Tekken 3's axes to Tekken Tag's (W_ttt = W_t3 · D; measured on the 14 fighters both
/// games have: OpenKumite's kumite_render/tests/crossover.rs): root and clavicles none, the feet half a turn about
/// y, the hands a quarter about x, the other animated joints half a turn about x; other nodes none.
fn turn(joint: &str) -> Q {
    let h = std::f64::consts::FRAC_1_SQRT_2;
    match joint {
        "foot_A" | "foot_B" => [0.0, 1.0, 0.0, 0.0],
        "hand_A" | "hand_B" => [-h, 0.0, 0.0, h],
        "chest" | "pelvis" | "thigh_A" | "shin_A" | "thigh_B" | "shin_B" | "upperarm_A" | "forearm_A"
        | "upperarm_B" | "forearm_B" | "head" => [1.0, 0.0, 0.0, 0.0],
        _ => [0.0, 0.0, 0.0, 1.0],
    }
}

/// A glb's JSON and binary chunks.
fn glb_split(b: &[u8]) -> Result<(Value, Vec<u8>)> {
    let u32at = |o: usize| b.get(o..o + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])).context("a short glb");
    if u32at(0)? != 0x46546C67 {
        bail!("not a glb");
    }
    let jl = u32at(12)? as usize;
    let j = serde_json::from_slice(b.get(20..20 + jl).context("a short glb")?)?;
    let bl = u32at(20 + jl)? as usize;
    let bin = b.get(28 + jl..28 + jl + bl).context("a short glb")?.to_vec();
    Ok((j, bin))
}

fn glb_join(j: &Value, bin: &[u8]) -> Result<Vec<u8>> {
    let mut js = serde_json::to_vec(j)?;
    while !js.len().is_multiple_of(4) {
        js.push(b' ');
    }
    let mut out = Vec::with_capacity(28 + js.len() + bin.len());
    for w in [0x46546C67, 2, (28 + js.len() + bin.len()) as u32, js.len() as u32, 0x4E4F534A] {
        out.extend(u32::to_le_bytes(w));
    }
    out.extend(js);
    out.extend((bin.len() as u32).to_le_bytes());
    out.extend(0x004E4942u32.to_le_bytes());
    out.extend(bin);
    Ok(out)
}

/// An accessor's components as floats, and where each lies in the binary chunk (no strides: this converter's glbs).
fn accessor(j: &Value, bin: &[u8], k: usize) -> Result<(Vec<f64>, Vec<usize>, u64, usize)> {
    let a = &j["accessors"][k];
    let view = &j["bufferViews"][n(&a["bufferView"]).context("an accessor without a view")?];
    if view.get("byteStride").is_some() {
        bail!("accessor {k}: a strided view");
    }
    let width = match a["type"].as_str() {
        Some("SCALAR") => 1,
        Some("VEC3") => 3,
        Some("VEC4") => 4,
        Some("MAT4") => 16,
        t => bail!("accessor {k}: type {t:?}"),
    };
    let ct = a["componentType"].as_u64().unwrap_or(0);
    let size = match ct {
        5126 => 4,
        5122 => 2,
        _ => bail!("accessor {k}: component type {ct}"),
    };
    let start = n(&view["byteOffset"]).unwrap_or(0) + n(&a["byteOffset"]).unwrap_or(0);
    let count = n(&a["count"]).context("an accessor without a count")? * width;
    let mut vals = Vec::with_capacity(count);
    let mut at = Vec::with_capacity(count);
    for i in 0..count {
        let o = start + i * size;
        let s = bin.get(o..o + size).with_context(|| format!("accessor {k} past the binary chunk"))?;
        vals.push(if size == 4 {
            f32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f64
        } else {
            (i16::from_le_bytes([s[0], s[1]]) as f64 / 32767.0).max(-1.0)
        });
        at.push(o);
    }
    Ok((vals, at, ct, width))
}

fn put(bin: &mut [u8], at: &[usize], ct: u64, vals: &[f64]) {
    for (&o, &v) in at.iter().zip(vals) {
        if ct == 5126 {
            bin[o..o + 4].copy_from_slice(&(v as f32).to_le_bytes());
        } else {
            bin[o..o + 2].copy_from_slice(&((v.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
        }
    }
}

/// A Tekken 3 character glb with each joint's axes turned onto Tekken Tag's ([`turn`]): its nodes' rest rotations
/// and translations, its inverse bind matrices and its motions (L' = D_parent⁻¹ · L · D, t' = D_parent⁻¹ · t); the
/// mesh as it is, every joint where it was.
pub fn retarget_glb(b: &[u8]) -> Result<Vec<u8>> {
    let (mut j, mut bin) = glb_split(b)?;
    let nodes = arr(&j["nodes"]).to_vec();
    let mut parent = vec![None; nodes.len()];
    for (p, node) in nodes.iter().enumerate() {
        for c in arr(&node["children"]).iter().filter_map(n) {
            *parent.get_mut(c).context("a child past the nodes")? = Some(p);
        }
    }
    let d: Vec<Q> = nodes.iter().map(|node| turn(node["name"].as_str().unwrap_or(""))).collect();
    let dp: Vec<Q> = parent.iter().map(|p| p.map_or([0.0, 0.0, 0.0, 1.0], |p| d[p])).collect();
    let num = |x: f64| serde_json::Number::from_f64(x).map_or(Value::Null, Value::Number);
    for (i, node) in j["nodes"].as_array_mut().into_iter().flatten().enumerate() {
        let Some(node) = node.as_object_mut() else { continue };
        if let Some(r) = node.get("rotation").and_then(|r| r.as_array()).filter(|r| r.len() == 4) {
            let q: Q = std::array::from_fn(|k| r[k].as_f64().unwrap_or(0.0));
            let q = qmul(qmul(conj(dp[i]), q), d[i]);
            node.insert("rotation".into(), Value::Array(q.iter().map(|&x| num(x)).collect()));
        } else if d[i] != [0.0, 0.0, 0.0, 1.0] || dp[i] != [0.0, 0.0, 0.0, 1.0] {
            node.insert("rotation".into(), Value::Array(qmul(conj(dp[i]), d[i]).iter().map(|&x| num(x)).collect()));
        }
        if let Some(t) = node.get("translation").and_then(|t| t.as_array()).filter(|t| t.len() == 3) {
            let v = rotate(conj(dp[i]), std::array::from_fn(|k| t[k].as_f64().unwrap_or(0.0)));
            node.insert("translation".into(), Value::Array(v.iter().map(|&x| num(x)).collect()));
        }
    }
    // the inverse bind matrices (column-major): R(D)⁻¹ · IBM
    for skin in arr(&j["skins"]).to_vec() {
        let Some(k) = n(&skin["inverseBindMatrices"]) else { continue };
        let (mut m, at, ct, _) = accessor(&j, &bin, k)?;
        for (s, joint) in arr(&skin["joints"]).iter().filter_map(n).enumerate() {
            let r = qmat(conj(*d.get(joint).context("a joint past the nodes")?));
            let Some(mat) = m.get_mut(s * 16..s * 16 + 16) else { bail!("too few inverse bind matrices") };
            for c in 0..4 {
                let col = [mat[c * 4], mat[c * 4 + 1], mat[c * 4 + 2]];
                for (row, rr) in r.iter().enumerate() {
                    mat[c * 4 + row] = rr[0] * col[0] + rr[1] * col[1] + rr[2] * col[2];
                }
            }
        }
        put(&mut bin, &at, ct, &m);
    }
    // the motions (an output turned once, refused when two joints of other turns share it)
    let mut done: BTreeMap<usize, (Q, Q, bool)> = BTreeMap::new();
    for anim in arr(&j["animations"]) {
        for ch in arr(&anim["channels"]) {
            let (Some(node), Some(path)) = (n(&ch["target"]["node"]), ch["target"]["path"].as_str()) else { continue };
            let Some(out) = n(&ch["sampler"]).and_then(|s| n(&anim["samplers"][s]["output"])) else { continue };
            let rot = match path {
                "rotation" => true,
                "translation" => false,
                _ => continue,
            };
            let key = (*dp.get(node).context("a channel past the nodes")?, d[node], rot);
            if let Some(k) = done.get(&out) {
                if *k != key {
                    bail!("accessor {out} is two joints' motion");
                }
                continue;
            }
            done.insert(out, key);
            let (mut v, at, ct, width) = accessor(&j, &bin, out)?;
            for e in v.chunks_mut(width) {
                if rot && width == 4 {
                    let q = qmul(qmul(conj(key.0), [e[0], e[1], e[2], e[3]]), key.1);
                    e.copy_from_slice(&q);
                } else if !rot && width == 3 {
                    e.copy_from_slice(&rotate(conj(key.0), [e[0], e[1], e[2]]));
                }
            }
            put(&mut bin, &at, ct, &v);
        }
    }
    glb_join(&j, &bin)
}

// ---------------------------------------------------------------------------------------------------------------
// The pairing

/// Both games' data the pairing and the movesets read.
struct Data {
    /// Tekken 3: the common set, the shared lists and sequences, the reactions, the roster.
    common: Value,
    shared: Value,
    reactions: Value,
    roster: Value,
    /// Tekken 3's movesets by file name (those read).
    sets: BTreeMap<String, Value>,
    /// Tekken Tag: the move table, the cancels, the hit records, the reactions, the roster, the moveset index.
    moves: Vec<Value>,
    cancels: Value,
    hits: Vec<Value>,
    ttt_reactions: Value,
    ttt_roster: Value,
    index: Value,
}

impl Data {
    fn load(t3: &dyn Vfs, ttt: &dyn Vfs) -> Result<Self> {
        let moves: Value = read(ttt, "gamedata/moves.json")?;
        let hits: Value = read(ttt, "gamedata/hits.json")?;
        let roster = read(t3, "gamedata/roster.json")?;
        let mut sets = BTreeMap::new();
        for e in arr(&roster["selections"]["entries"]) {
            let wanted = |c: u64| (c as usize) < SHARED || FIGHTERS.iter().any(|f| f.0 == c as usize);
            if let Some(f) = e["moveset_file"].as_str().filter(|_| e["character"].as_u64().is_some_and(wanted))
                && !sets.contains_key(f)
            {
                sets.insert(f.to_string(), read(t3, &format!("gamedata/{f}"))?);
            }
        }
        Ok(Self {
            common: read(t3, "gamedata/movesets/common.json")?,
            shared: read(t3, "gamedata/shared.json")?,
            reactions: read(t3, "gamedata/reactions.json")?,
            roster,
            sets,
            moves: arr(&moves["moves"]).to_vec(),
            cancels: read(ttt, "gamedata/cancels.json")?,
            hits: arr(&hits["records"]).to_vec(),
            ttt_reactions: read(ttt, "gamedata/reactions.json")?,
            ttt_roster: read(ttt, "gamedata/roster.json")?,
            index: read(ttt, "gamedata/movesets/index.json")?,
        })
    }

    /// Tekken 3's moveset of character `c` (its first costume's).
    fn t3_set(&self, c: usize) -> Option<&Value> {
        let e = arr(&self.roster["selections"]["entries"])
            .iter()
            .find(|e| n(&e["character"]) == Some(c) && e["moveset_file"].is_string())?;
        self.sets.get(e["moveset_file"].as_str()?)
    }

    fn t3_name(&self, c: usize) -> Option<String> {
        arr(&self.roster["characters"])
            .iter()
            .find(|e| n(&e["id"]) == Some(c))
            .and_then(|e| e["name"].as_str())
            .map(short)
    }

    /// Tekken Tag's body of a name, and its moveset id.
    fn ttt_body(&self, name: &str) -> Option<(usize, usize)> {
        arr(&self.ttt_roster["characters"])
            .iter()
            .find(|e| e["name"].as_str().map(short).as_deref() == Some(name))
            .and_then(|e| Some((n(&e["id"])?, n(&e["moveset_id"])?)))
    }

    /// Tekken 3's move of alias `a` in `set` (its own or a common one).
    fn t3_move<'a>(&'a self, set: &'a Value, a: usize) -> Option<&'a Value> {
        let r = set["aliases"].get(a)?;
        let i = n(&r["move"])?;
        match r["bank"].as_str()? {
            "own" => set["moves"].get(i),
            "common" => self.common["moves"].get(i),
            _ => None,
        }
    }

    /// A Tekken 3 move's command list (from `start` in `cancels`), the shared lists it calls inlined, to its END.
    fn t3_list<'a>(&'a self, cancels: &'a Value, start: usize) -> Vec<&'a Value> {
        let shared = arr(&self.shared["shared_cancels"]["entries"]);
        let mut out = Vec::new();
        for c in arr(cancels).iter().skip(start) {
            match c["kind"].as_str() {
                Some("end") => {
                    out.push(c);
                    break;
                }
                Some("call") => out.extend(
                    shared.iter().skip(n(&c["dest"]).unwrap_or(usize::MAX)).take_while(|e| e["kind"] != "return"),
                ),
                _ => out.push(c),
            }
        }
        out
    }

    fn t3_cancels<'a>(&'a self, set: &'a Value, m: &'a Value, own: bool) -> Vec<&'a Value> {
        self.t3_list(if own { &set["cancels"] } else { &self.common["cancels"] }, n(&m["cancel_start"]).unwrap_or(0))
    }

    fn ttt_cancels(&self, m: &Value) -> Vec<&Value> {
        let shared = arr(&self.cancels["shared_cancels"]["entries"]);
        let mut out = Vec::new();
        for c in arr(&self.cancels["cancels"]["entries"]).iter().skip(n(&m["cancel_start"]).unwrap_or(0)) {
            match c["kind"].as_str() {
                Some("end") => {
                    out.push(c);
                    break;
                }
                Some("call") => out.extend(
                    shared.iter().skip(n(&c["dest"]).unwrap_or(usize::MAX)).take_while(|e| e["kind"] != "return"),
                ),
                _ => out.push(c),
            }
        }
        out
    }

    /// A Tekken Tag move's reaction: its first hit record without a gate.
    fn ttt_reaction(&self, m: &Value) -> Option<usize> {
        self.hits.get(n(&m["hit"]).unwrap_or(0)..)?.iter().find(|h| h["gate"].is_null()).and_then(|h| n(&h["reaction"]))
    }

    /// Tekken Tag's aliases of each moveset file.
    fn ttt_aliases(&self, ttt: &dyn Vfs) -> Result<BTreeMap<String, Vec<Option<usize>>>> {
        let mut out = BTreeMap::new();
        for f in arr(&self.index["fighters"]).iter().filter_map(|f| f["moveset_file"].as_str()) {
            if !out.contains_key(f) {
                let set = read(ttt, &format!("gamedata/movesets/{f}"))?;
                out.insert(f.to_string(), arr(&set["aliases"]).iter().map(n).collect());
            }
        }
        Ok(out)
    }
}

/// Tekken 3's aliases and reactions with the Tekken Tag ones they pair with, and how many votes each got.
#[derive(Default)]
pub struct Pairing {
    pub alias: BTreeMap<usize, usize>,
    pub reaction: BTreeMap<usize, usize>,
}

fn key(m: &Value) -> (u64, u64, u64, u64) {
    let a = |k: usize| m["active"][k].as_u64().unwrap_or(0);
    (m["frames"].as_u64().unwrap_or(0), m["attack_mask"]["raw"].as_u64().unwrap_or(0) & 0xFFFF, a(0), a(1))
}

fn attack(m: &Value) -> bool {
    m["attack_mask"]["raw"].as_u64().unwrap_or(0) != 0 && m["active"][0].as_u64().unwrap_or(0) != 0
}

/// Pairs Tekken 3's aliases and reactions with Tekken Tag's (the module's notes); `aliases`: Tekken Tag's
/// movesets' aliases by file.
fn pairing(d: &Data, aliases: &BTreeMap<String, Vec<Option<usize>>>) -> Pairing {
    let mut votes: BTreeMap<(usize, usize), u32> = BTreeMap::new();
    let mut rvotes: BTreeMap<(usize, usize), u32> = BTreeMap::new();
    let mut fighters = Vec::new();
    for f in 0..SHARED {
        let (Some(set), Some(name)) = (d.t3_set(f), d.t3_name(f)) else { continue };
        let Some((body, _)) = d.ttt_body(&name) else { continue };
        let Some(tt) = arr(&d.index["fighters"])
            .iter()
            .find(|e| n(&e["body"]) == Some(body))
            .and_then(|e| aliases.get(e["moveset_file"].as_str()?))
        else {
            continue;
        };
        fighters.push((set, tt));
        // each move's aliases on both sides; the attacks by their keys
        let mut m3: BTreeMap<(bool, usize), Vec<usize>> = BTreeMap::new();
        for (a, r) in arr(&set["aliases"]).iter().enumerate() {
            if let Some(i) = n(&r["move"]) {
                m3.entry((r["bank"] == "own", i)).or_default().push(a);
            }
        }
        let mut mt: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (a, i) in tt.iter().enumerate() {
            if let Some(i) = i {
                mt.entry(*i).or_default().push(a);
            }
        }
        let mut k3: BTreeMap<_, Vec<(bool, usize)>> = BTreeMap::new();
        for &(own, i) in m3.keys() {
            if let Some(m) = (if own { &set["moves"] } else { &d.common["moves"] }).get(i).filter(|m| attack(m)) {
                k3.entry(key(m)).or_default().push((own, i));
            }
        }
        let mut kt: BTreeMap<_, Vec<usize>> = BTreeMap::new();
        for &i in mt.keys() {
            if let Some(m) = d.moves.get(i).filter(|m| attack(m)) {
                kt.entry(key(m)).or_default().push(i);
            }
        }
        for (k, l) in &k3 {
            let (&[m], Some(&[t])) = (l.as_slice(), kt.get(k).map(|v| v.as_slice())) else { continue };
            for &a in &m3[&m] {
                for &b in &mt[&t] {
                    *votes.entry((a, b)).or_default() += 1;
                }
            }
            let mv = if m.0 { &set["moves"][m.1] } else { &d.common["moves"][m.1] };
            if let (Some(r), Some(rt)) = (
                n(&mv["reaction"]["index"]).filter(|_| mv["reaction"]["table"] == "reactions"),
                d.moves.get(t).and_then(|x| d.ttt_reaction(x)),
            ) {
                *rvotes.entry((r, rt)).or_default() += 1;
            }
        }
    }
    let mut reaction: BTreeMap<usize, (u32, usize)> = BTreeMap::new();
    for (&(a, b), &v) in &rvotes {
        let e = reaction.entry(a).or_insert((v, b));
        if v > e.0 {
            *e = (v, b);
        }
    }
    let reaction: BTreeMap<usize, usize> = reaction.into_iter().map(|(a, (_, b))| (a, b)).collect();
    let (r3, rt) = (arr(&d.reactions["entries"]), arr(&d.ttt_reactions["entries"]));
    for (&a, &b) in &reaction {
        for s in 0..14 {
            let (Some(v), Some(w)) =
                (r3.get(a).and_then(|e| n(&e["victims"][s])), rt.get(b).and_then(|e| n(&e["victims"][s])))
            else {
                continue;
            };
            if v != 0 && w != 0 {
                *votes.entry((v, w)).or_default() += 3;
            }
        }
    }
    let (air3, airt) =
        (arr(&d.reactions["air_reactions"]["entries"]), arr(&d.ttt_reactions["air_reactions"]["entries"]));
    for (a, b) in AIR.into_iter().take(6) {
        if let (Some(v), Some(w)) = (air3.get(a).and_then(|e| n(&e["alias"])), airt.get(b).and_then(|e| n(&e["alias"])))
        {
            *votes.entry((v, w)).or_default() += 10;
        }
    }
    for (a, b) in FIXED {
        *votes.entry((a, b)).or_default() += 100;
    }
    // the paired moves' next moves, and their commands' destinations (a command both lists have once)
    for _ in 0..3 {
        let now: Vec<(usize, usize)> = votes.iter().filter(|e| *e.1 >= 2).map(|e| *e.0).collect();
        for (a, b) in now {
            for (set, tt) in &fighters {
                let Some(x) = d.t3_move(set, a) else { continue };
                let Some(y) = tt.get(b).copied().flatten().and_then(|i| d.moves.get(i)) else { continue };
                if x["frames"] != y["frames"] {
                    continue;
                }
                if let (Some(p), Some(q)) = (n(&x["next"]["alias"]), n(&y["next"]["alias"])) {
                    *votes.entry((p, q)).or_default() += 1;
                }
                let own = set["aliases"][a]["bank"] == "own";
                let by = |l: Vec<&Value>| {
                    let mut m: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
                    for c in l {
                        if let (Some(k), Some(dest)) = (
                            c["kind"]
                                .as_str()
                                .filter(|k| ["command", "sequence", "dash", "backdash", "end"].contains(k)),
                            n(&c["dest"]),
                        ) {
                            m.entry((k.to_string(), c["input"].as_str().unwrap_or("").to_string()))
                                .or_default()
                                .push(dest);
                        }
                    }
                    m
                };
                let (l3, lt) = (by(d.t3_cancels(set, x, own)), by(d.ttt_cancels(y)));
                for (k, v) in &l3 {
                    if let (&[p], Some(&[q])) = (v.as_slice(), lt.get(k).map(|v| v.as_slice())) {
                        *votes.entry((p, q)).or_default() += 1;
                    }
                }
            }
        }
    }
    let mut ranked: Vec<_> = votes.into_iter().filter(|e| e.1 >= 2).collect();
    ranked.sort_by_key(|&((a, b), v)| (std::cmp::Reverse(v), a, b));
    let (mut alias, mut used) = (BTreeMap::new(), BTreeSet::new());
    for ((a, b), _) in ranked {
        if b < TTT_ALIASES && !alias.contains_key(&a) && used.insert(b) {
            alias.insert(a, b);
        }
    }
    Pairing { alias, reaction }
}

// ---------------------------------------------------------------------------------------------------------------
// The movesets

/// How the crossover's aliases and reactions found their Tekken Tag counterparts (the install's log).
#[derive(Default, Debug)]
pub struct Coverage {
    /// Per fighter: its aliases, those paired with a Tekken Tag alias (the rest numbered from 5532 on).
    pub aliases: Vec<(usize, usize)>,
    /// Per fighter: the reactions its attacks deal, those paired with a Tekken Tag reaction (the rest added).
    pub dealt: Vec<(usize, usize)>,
    /// Per fighter: Tekken Tag's victim aliases (every reaction's 14 slots, the air table's), filled by an alias pair,
    /// by the paired reaction's slot, and left to the opponent's entry (Tekken 3's merged alias table).
    pub received: Vec<[usize; 3]>,
    /// The air table's aliases filled, of its aliases.
    pub air: (usize, usize),
    /// Tekken 3's motion inputs the crossover uses without a Tekken Tag sequence of the same steps.
    pub sequences_unpaired: usize,
}

/// The numbers the two fighters' aliases take in Tekken Tag (one numbering for both: one's reaction names the
/// other's victim move as Tekken 3's aliases do), and the reactions added to Tekken Tag's table.
struct Space<'a> {
    d: &'a Data,
    p: &'a Pairing,
    num: BTreeMap<usize, usize>,
    added: Vec<Value>,
    made: BTreeMap<(bool, usize), usize>,
    /// Each fighter's Tekken 3 sound scripts (character, script) as Tekken Tag's ([`scripts`]).
    scripts: BTreeMap<(usize, usize), usize>,
    /// Tekken Tag's aliases whose list has a plain tag command (no gate: the fallback after the screen sides'), to its
    /// destination: the stance's and the crouch's run off (5394), lying face up's and down's roll offs (5433, 5438).
    tags: BTreeMap<usize, usize>,
    /// Tekken Tag's entrance move (alias 5397: the run in).
    run_in: Option<Value>,
}

impl Space<'_> {
    fn alias(&self, v: usize) -> usize {
        self.num.get(&v).copied().unwrap_or(usize::MAX)
    }

    /// Tekken Tag's reaction for a Tekken 3 move's: the paired one, else one added (a throw's: its victim alias).
    fn reaction(&mut self, r: &Value) -> Value {
        let (throw, k) = (r["table"] == "throw_victims", n(&r["index"]).unwrap_or(0));
        if !throw && k == 0 {
            return json!({"index": 0, "table": "reactions"});
        }
        if let Some(&b) = self.p.reaction.get(&k).filter(|_| !throw) {
            return json!({"index": b, "table": "reactions"});
        }
        let base = arr(&self.d.ttt_reactions["entries"]).len();
        let id = match self.made.get(&(throw, k)) {
            Some(&id) => id,
            None => {
                let id = base + self.added.len();
                let e = if throw {
                    let v = arr(&self.d.reactions["throw_victims"]["entries"])
                        .get(k)
                        .and_then(n)
                        .map_or(0, |v| self.alias(v));
                    json!({"index": id, "victims": vec![v; 14], "params": [0, null, 0, null, 0, null, null]})
                } else {
                    let e = &self.d.reactions["entries"][k];
                    let victims: Vec<usize> = arr(&e["victims"])
                        .iter()
                        .map(|v| n(v).filter(|&v| v != 0).map_or(0, |v| self.alias(v)))
                        .collect();
                    let params: Vec<Value> = arr(&e["params"])
                        .iter()
                        .enumerate()
                        .map(|(i, p)| if [1, 3, 5, 6].contains(&i) { self.curve(p) } else { p.clone() })
                        .collect();
                    json!({"index": id, "victims": victims, "params": params})
                };
                self.added.push(e);
                self.made.insert((throw, k), id);
                id
            }
        };
        json!({"index": id, "table": if throw { "throw_victims" } else { "reactions" }})
    }

    /// Tekken Tag's pushback curve (its entry) of Tekken 3's curve reference (its multiple of 10): the same curve,
    /// else the nearest.
    fn curve(&self, p: &Value) -> Value {
        let Some(c3) = arr(&self.d.reactions["pushback_curves"]["entries"]).iter().find(|c| c["index"] == *p) else {
            return Value::Null;
        };
        let samples = |c: &Value| -> Vec<i64> {
            let mut v = vec![c["constant_frames"].as_i64().unwrap_or(0), c["velocity"].as_i64().unwrap_or(0)];
            v.extend((0..10).map(|k| c["impulses"][k].as_i64().unwrap_or(0)));
            v
        };
        let a = samples(c3);
        arr(&self.d.ttt_reactions["pushback_curves"]["entries"])
            .iter()
            .min_by_key(|c| {
                (samples(c).iter().zip(&a).map(|(x, y)| (x - y).abs()).sum::<i64>(), c["index"].as_i64().unwrap_or(0))
            })
            .map_or(Value::Null, |c| c["index"].clone())
    }
}

/// Tekken 3's moveset id to Tekken Tag's, and character id to body (by name; the crossover's own).
fn ids(d: &Data) -> (BTreeMap<usize, usize>, BTreeMap<usize, usize>) {
    let (mut ms, mut body) = (BTreeMap::new(), BTreeMap::new());
    for e in arr(&d.roster["selections"]["entries"]) {
        let (Some(c), Some(m)) = (n(&e["character"]), n(&e["moveset_id"])) else { continue };
        let to = FIGHTERS.iter().find(|f| f.0 == c).map(|f| (f.1, f.2)).or_else(|| d.ttt_body(&d.t3_name(c)?));
        if let Some((b, s)) = to {
            ms.entry(m).or_insert(s);
            body.insert(c, b);
        }
    }
    (ms, body)
}

/// A Tekken 3 `req` gate in the engine's widened form (kumite_sim::dialect): without a Tekken Tag counterpart the
/// "is" kinds never pass, the "not" kinds always.
fn req(r: u64, ms: &BTreeMap<usize, usize>, body: &BTreeMap<usize, usize>) -> u64 {
    let r = r as usize;
    let (kind, map, v, not) = match r {
        0 => return 0,
        0x01..=0x17 => (1, ms, r - 1, false),
        0x18..=0x2E => (2, ms, r - 0x18, true),
        0x2F..=0x45 => (3, ms, r - 0x2F, false),
        0x46..=0x5C => (4, body, r - 0x46, false),
        _ => (5, body, r - 0x5D, true),
    };
    match map.get(&v) {
        Some(&x) => (kind << 8 | x) as u64,
        None if not => 0,
        None => 0xFF << 8,
    }
}

/// The aliases Tekken 3's fighter `set` plays: its own entries, and the common moves the fighters both games have
/// give an alias it lacks (Tekken 3 takes those from the opponent's set). Alias -> (own move?, index).
fn entries(d: &Data, set: &Value) -> BTreeMap<usize, (bool, usize)> {
    let mut e = BTreeMap::new();
    let mut lacks = BTreeSet::new();
    for (a, r) in arr(&set["aliases"]).iter().enumerate() {
        match (r["bank"].as_str(), n(&r["move"])) {
            (Some("own"), Some(i)) => _ = e.insert(a, (true, i)),
            (Some("common"), Some(i)) => _ = e.insert(a, (false, i)),
            _ => _ = lacks.insert(a),
        }
    }
    for a in lacks {
        let mut count: BTreeMap<usize, u32> = BTreeMap::new();
        for f in 0..SHARED {
            if let Some(i) = d
                .t3_set(f)
                .and_then(|s| s["aliases"].get(a))
                .filter(|r| r["bank"] == "common")
                .and_then(|r| n(&r["move"]))
            {
                *count.entry(i).or_default() += 1;
            }
        }
        if let Some((&i, _)) = count.iter().max_by_key(|&(&i, &c)| (c, std::cmp::Reverse(i))) {
            e.insert(a, (false, i));
        }
    }
    e
}

/// Tekken Tag's motion input of the same steps as Tekken 3's command `cmd`.
fn sequence(d: &Data, cmd: u64) -> Option<u64> {
    let s3 = arr(&d.shared["sequences"]["entries"]).iter().find(|s| s["cmd"].as_u64() == Some(cmd))?;
    let raw = |s: &Value| (s["window"].clone(), arr(&s["steps"]).iter().map(|t| t["raw"].as_u64()).collect::<Vec<_>>());
    arr(&d.cancels["sequences"]["entries"]).iter().find(|s| raw(s) == raw(s3)).and_then(|s| s["cmd"].as_u64())
}

type Ids = (BTreeMap<usize, usize>, BTreeMap<usize, usize>);

/// Fighter `c`'s moveset in Tekken Tag's alias space (its moves, the common ones it plays copied in, one command
/// list each with the shared lists inlined), with `id` its moveset id.
fn moveset(sp: &mut Space, c: usize, id: usize, ids: &Ids, cov: &mut Coverage) -> Result<Value> {
    let d = sp.d;
    let set = d.t3_set(c).with_context(|| format!("Tekken 3's character {c}: no moveset"))?;
    let e = entries(d, set);
    let own = arr(&set["moves"]);
    let ns = own.len();
    let all: Vec<(bool, &Value)> =
        own.iter().map(|m| (true, m)).chain(arr(&d.common["moves"]).iter().map(|m| (false, m))).collect();
    let index = |(o, i): (bool, usize)| if o { i } else { ns + i };
    let mut cancels: Vec<Value> = Vec::new();
    let mut starts: BTreeMap<(bool, usize), usize> = BTreeMap::new();
    let mut moves = Vec::with_capacity(all.len());
    for (k, &(o, m)) in all.iter().enumerate() {
        let start = n(&m["cancel_start"]).unwrap_or(0);
        let at = match starts.get(&(o, start)) {
            Some(&at) => at,
            None => {
                let at = cancels.len();
                for c in d.t3_list(if o { &set["cancels"] } else { &d.common["cancels"] }, start) {
                    cancels.push(command(sp, c, ids, cov));
                }
                starts.insert((o, start), at);
                at
            }
        };
        let mut m = m.clone();
        let Some(o) = m.as_object_mut() else { bail!("move {k}: not an object") };
        for key in ["raw_hex", "byte_2F", "motion"] {
            o.remove(key);
        }
        o.insert("index".into(), json!(k));
        o.insert("cancel_start".into(), json!(at));
        let next = n(&o["next"]["alias"]).unwrap_or(0);
        let hi = o["next"]["hi"].clone();
        o.insert("next".into(), json!({"alias": sp.alias(next), "hi": hi}));
        let r = o["reaction"].clone();
        o.insert("reaction".into(), sp.reaction(&r));
        // shortcut: Tekken 3's close-range reactions (2 of Gon's moves, 10 of Doctor B.'s) are dropped: Tekken Tag's
        // are hit records; add one to hits.json if a close hit matters
        o.insert("close_range".into(), json!(0));
        // (Tekken Tag's reversal families hold its motion numbers: no Tekken 3 tag is one)
        o.insert("motion_tag".into(), json!(-1));
        o.insert("air_reaction".into(), json!(0));
        // the fight camera's yaw caps: Tekken 3's camera ids 0x27-0x2A are Tekken Tag's 0x13-0x16
        let cam = o["flags24"]["camera_script"].as_u64().unwrap_or(0);
        o["flags24"]["camera_script"] = json!(if (0x27..=0x2A).contains(&cam) { cam - 0x14 } else { 0 });
        let sounds = move_sounds(&o["sounds"], |i| sp.scripts.get(&(c, i)).copied());
        o.insert("sounds".into(), sounds);
        moves.push(m);
    }
    let extra = sp.num.values().filter(|&&v| v >= TTT_ALIASES).count();
    let mut aliases: Vec<Option<usize>> = vec![None; TTT_ALIASES + extra];
    for (&a, &mv) in &e {
        if let Some(slot) = aliases.get_mut(sp.alias(a)) {
            *slot = Some(index(mv));
        }
    }
    let [stance, back, start] = [3, 30, 2].map(|a| e.get(&a).map(|&mv| index(mv)));
    let (Some(stance), Some(back), Some(start)) = (stance, back, start) else {
        bail!("Tekken 3's character {c}: no stance, backdash or round start")
    };
    // (the entrance ends in the stance; lying (Doctor B.'s), in the round start by × instead, standing to his guard)
    let lying = arr(&moves[stance]["posture"]["names"]).contains(&json!("lying"));
    let after = if lying { sp.alias(2) } else { 58 };
    // the moves Tekken Tag tags from: those whose paired alias's list has a plain tag command, to its destination
    let mut tagged: BTreeMap<usize, usize> = BTreeMap::new();
    for (a, &mv) in &e {
        if let Some(&dest) = sp.p.alias.get(a).and_then(|b| sp.tags.get(b)) {
            tagged.entry(index(mv)).or_insert(dest);
        }
    }
    let run_in = sp.run_in.as_ref().context("Tekken Tag has no entrance")?;
    tag_moves(&mut moves, &mut cancels, &mut aliases, &tagged, [back, start], (run_in, after));
    cov.aliases.push((e.len(), e.keys().filter(|a| sp.p.alias.contains_key(a)).count()));
    let dealt: BTreeSet<_> = own
        .iter()
        .filter(|m| attack(m) && m["reaction"]["table"] == "reactions")
        .filter_map(|m| n(&m["reaction"]["index"]))
        .collect();
    cov.dealt.push((dealt.len(), dealt.iter().filter(|r| sp.p.reaction.contains_key(r)).count()));
    // the victim aliases of Tekken Tag's hits (not its throws': those are the thrower's) this fighter lacks: by the
    // alias pair, else by the paired reaction's slot; the rest are the opponent's (resolved in the fight, as Tekken
    // 3's merged alias table does)
    let hits: BTreeSet<usize> = d
        .moves
        .iter()
        .filter(|m| attack(m) && !matches!(m["attack_mask"]["level"].as_u64(), Some(7 | 8)))
        .filter_map(|m| d.ttt_reaction(m))
        .collect();
    let inv: BTreeMap<usize, usize> = sp.p.alias.iter().map(|(&a, &b)| (b, a)).collect();
    let mut rinv: BTreeMap<usize, usize> = BTreeMap::new();
    for (&a, &b) in &sp.p.reaction {
        rinv.entry(b).or_insert(a);
    }
    let mut victims: Vec<(usize, Option<usize>)> = Vec::new();
    for (r, entry) in arr(&d.ttt_reactions["entries"]).iter().enumerate().filter(|(r, _)| hits.contains(r)) {
        for (s, w) in arr(&entry["victims"]).iter().enumerate() {
            let by_slot = rinv.get(&r).and_then(|&r3| n(&d.reactions["entries"][r3]["victims"][s]));
            victims.extend(n(w).filter(|&w| w != 0).map(|w| (w, by_slot)));
        }
    }
    let (air3, airt) =
        (arr(&d.reactions["air_reactions"]["entries"]), arr(&d.ttt_reactions["air_reactions"]["entries"]));
    for (a, b) in AIR {
        if let Some(w) = airt.get(b).and_then(|x| n(&x["alias"])) {
            victims.push((w, air3.get(a).and_then(|x| n(&x["alias"]))));
        }
    }
    let (mut got, mut seen) = ([0; 3], BTreeSet::new());
    for (w, by_slot) in victims {
        if w >= aliases.len() || !seen.insert(w) {
            continue;
        }
        if aliases[w].is_some() {
            got[0] += 1;
        } else if let Some(&mv) = inv.get(&w).and_then(|v| e.get(v)) {
            aliases[w] = Some(index(mv));
            got[0] += 1;
        } else if let Some(&mv) = by_slot.and_then(|v| e.get(&v)) {
            aliases[w] = Some(index(mv));
            got[1] += 1;
        } else {
            got[2] += 1;
        }
    }
    cov.received.push(got);
    let air: BTreeSet<usize> = airt.iter().filter_map(|x| n(&x["alias"])).collect();
    cov.air = (air.iter().filter(|&&w| aliases.get(w).is_some_and(|a| a.is_some())).count(), air.len());
    let aliases: Vec<Value> = aliases
        .into_iter()
        .map(|a| a.map_or(json!({"bank": "none", "move": null}), |i| json!({"bank": "own", "move": i})))
        .collect();
    Ok(
        json!({"moveset_id": id, "moves": moves, "cancels": cancels, "aliases": aliases, "camera_motions": set["camera_motions"].clone()}),
    )
}

/// Tekken Tag's run off for a tag (aliases 5394, 5398: from the stance, from other moves) and entrance (5397).
const TAG: [usize; 3] = [5394, 5398, 5397];

/// The made-up tag moves (Tekken 3 has none): the tag button (command bit 14) in the moves `tagged` (`moves` index
/// to the alias it goes to: Tekken Tag's own for the state, [`Space::tags`]) starts that alias; the run offs, the
/// backdash (`moves` index `[back, intro]`; tagging: posture bit 24), swap the partner in once off the screen or at
/// its end (as Tekken Tag's own: the swap bit, facing the opponent); lying, Tekken Tag's own roll off (the opponent's
/// entry). The entrance is Tekken Tag's run in (`run_in`: its motion and frames, from off the screen) on the round
/// start by × (`start`), then alias `after`.
fn tag_moves(
    moves: &mut Vec<Value>,
    cancels: &mut Vec<Value>,
    aliases: &mut [Option<usize>],
    tagged: &BTreeMap<usize, usize>,
    [back, start]: [usize; 2],
    (run_in, after): (&Value, usize),
) {
    let entry = |kind: &str,
                 cmd: u32,
                 input: &str,
                 cond: u32,
                 dest: usize,
                 win: [u32; 2],
                 transition: u32,
                 tag: bool| {
        json!({"kind": kind, "cmd": cmd, "input": input, "req": 0, "cond": cond, "cond_tag": tag, "param": 0, "dest": dest, "win_start": win[0], "win_end": win[1], "exec": win[0], "transition": transition})
    };
    // (kumite_sim::dialect: the off-screen condition 0x101; the swap 0x100 with facing the opponent 0x2D and 0x40)
    let swap = 0x2D | 0x100 | 0x40;
    for (&k, &dest) in tagged {
        let (start, list) = (cancels.len(), moves[k]["cancel_start"].as_u64().unwrap_or(0) as usize);
        cancels.push(entry("command", 0x4000, "tag", 0, dest, [1, 255], 6, true));
        let end = cancels.iter().skip(list).position(|c| c["kind"] == "end").map_or(start, |i| list + i + 1);
        cancels.extend(cancels[list..end].to_vec());
        moves[k]["cancel_start"] = json!(start);
    }
    let mut made = |from: usize, list: Vec<Value>, next: usize| {
        let mut m = moves[from].clone();
        m["index"] = json!(moves.len());
        m["cancel_start"] = json!(cancels.len());
        m["next"] = json!({"alias": next, "hi": 0});
        cancels.extend(list);
        moves.push(m);
        moves.len() - 1
    };
    let run = made(
        back,
        vec![
            entry("command", 0x10, "*", 0x101, TAG[2], [1, 255], swap, false),
            entry("end", 0x8000, "END", 0, TAG[2], [0, 0], swap, false),
        ],
        TAG[2],
    );
    let enter = made(start, vec![entry("end", 0x8000, "END", 0, after, [0, 0], 6, false)], after);
    for k in ["anim", "frames"] {
        moves[enter][k] = run_in[k].clone();
    }
    let raw = moves[run]["posture"]["raw"].as_u64().unwrap_or(0);
    moves[run]["posture"]["raw"] = json!(raw | 1 << 24);
    for (a, mv) in TAG.into_iter().zip([run, run, enter]) {
        aliases[a] = Some(mv);
    }
}

/// A Tekken 3 command entry in Tekken Tag's numbers: its destination alias, its motion input, its gate; the reversal
/// of an alias takes Tekken Tag's alias, of a motion family never passes (Tekken Tag's families hold its motions).
fn command(sp: &Space, c: &Value, ids: &Ids, cov: &mut Coverage) -> Value {
    let mut c = c.clone();
    let kind = c["kind"].as_str().unwrap_or("").to_string();
    if ["command", "sequence", "dash", "backdash", "end"].contains(&kind.as_str()) {
        c["dest"] = json!(sp.alias(n(&c["dest"]).unwrap_or(0)));
    }
    if kind == "sequence" {
        match sequence(sp.d, c["cmd"].as_u64().unwrap_or(0)) {
            Some(s) => c["cmd"] = json!(s),
            None => {
                cov.sequences_unpaired += 1;
                c["cond"] = json!(0xFF);
            }
        }
    }
    c["req"] = json!(req(c["req"].as_u64().unwrap_or(0), &ids.0, &ids.1));
    match c["cond"].as_u64() {
        Some(0x4A) => c["param"] = json!(sp.alias(n(&c["param"]).unwrap_or(0))),
        Some(0x4B) => c["cond"] = json!(0xFF),
        _ => {}
    }
    c
}

/// The Tekken Tag motions a crossover fighter plays from its opponents' sets (Tekken Tag's own fighters list
/// theirs): what every Tekken Tag fighter's throws and reactions start on the victim, the aliases its own moves go to
/// that it lacks (`lacked`), and the moves they go on to.
fn motions(d: &Data, aliases: &BTreeMap<String, Vec<Option<usize>>>, lacked: &BTreeSet<usize>) -> Vec<String> {
    let links: BTreeSet<usize> = arr(&d.cancels["cancels"]["entries"])
        .iter()
        .chain(arr(&d.cancels["shared_cancels"]["entries"]))
        .filter(|c| matches!(c["transition"].as_u64().map(|t| t & 0x7F), Some(31 | 32)))
        .filter_map(|c| n(&c["dest"]))
        .collect();
    let reactions = arr(&d.ttt_reactions["entries"]);
    let mut out = BTreeSet::new();
    for tt in aliases.values() {
        let at = |a: usize| tt.get(a).copied().flatten().and_then(|i| d.moves.get(i));
        let mut todo: Vec<usize> = links
            .iter()
            .filter_map(|&a| at(a))
            .filter_map(|m| d.ttt_reaction(m))
            .filter_map(|r| reactions.get(r).and_then(|e| n(&e["victims"][0])))
            .collect();
        todo.extend(reactions.iter().flat_map(|e| arr(&e["victims"]).iter().filter_map(n)));
        todo.extend(arr(&d.ttt_reactions["air_reactions"]["entries"]).iter().filter_map(|e| n(&e["alias"])));
        todo.extend(lacked);
        let mut seen = BTreeSet::new();
        while let Some(a) = todo.pop() {
            if a == 0 || !seen.insert(a) {
                continue;
            }
            let Some(m) = at(a) else { continue };
            if let Some(anim) = m["anim"].as_str() {
                out.insert(format!("{anim}.glb"));
            }
            todo.extend(n(&m["next"]["alias"]));
            // (and its END: the moves it goes on to without an input)
            todo.extend(d.ttt_cancels(m).iter().filter(|c| c["kind"] == "end").filter_map(|c| n(&c["dest"])));
        }
    }
    out.into_iter().collect()
}

fn push(list: &mut Value, v: Value) -> Result<()> {
    list.as_array_mut().context("not a list")?.push(v);
    Ok(())
}

/// Installs Gon and Doctor B. into Tekken Tag Tournament: reads Tekken 3's game root `t3` and Tekken Tag's through
/// `ctx.out`, writes into `ctx.out` their bodies (characters/, turned onto Tekken Tag's axes), their movesets and
/// motion lists, and Tekken Tag's moveset index, roster, reactions and rules with them added.
pub fn install(ctx: &Ctx, t3: &dyn Vfs) -> Result<Coverage> {
    let ttt = ctx.out;
    let d = Data::load(t3, ttt)?;
    let aliases = d.ttt_aliases(ttt)?;
    let p = pairing(&d, &aliases);
    let ids = ids(&d);
    // one numbering for both fighters' aliases: the paired ones Tekken Tag's, the others from 5532 on
    let mut used = BTreeSet::new();
    for &(c, _, _) in &FIGHTERS {
        let set = d.t3_set(c).with_context(|| format!("Tekken 3's character {c}: no moveset"))?;
        used.extend(entries(&d, set).into_keys());
        used.extend(arr(&set["moves"]).iter().chain(arr(&d.common["moves"])).filter_map(|m| n(&m["next"]["alias"])));
        let lists = arr(&set["cancels"])
            .iter()
            .chain(arr(&d.common["cancels"]))
            .chain(arr(&d.shared["shared_cancels"]["entries"]));
        // (and the aliases a reversal names)
        used.extend(
            lists
                .flat_map(|c| {
                    [
                        (c["kind"] != "call").then(|| n(&c["dest"])).flatten(),
                        (c["cond"] == 0x4A).then(|| n(&c["param"])).flatten(),
                    ]
                })
                .flatten(),
        );
    }
    used.extend(arr(&d.reactions["entries"]).iter().flat_map(|e| arr(&e["victims"]).iter().filter_map(n)));
    used.extend(arr(&d.reactions["throw_victims"]["entries"]).iter().filter_map(n));
    let mut num = BTreeMap::new();
    let mut next = TTT_ALIASES;
    for a in used {
        let to = p.alias.get(&a).copied().unwrap_or_else(|| {
            next += 1;
            next - 1
        });
        num.insert(a, to);
    }
    let ttt_sound = read(ttt, "gamedata/sound.json")?;
    let mut scripts = BTreeMap::new();
    for &(c, _, _) in &FIGHTERS {
        let set = d.t3_set(c).with_context(|| format!("Tekken 3's character {c}: no moveset"))?;
        for i in script_ids(arr(&set["moves"]).iter().chain(arr(&d.common["moves"]))) {
            let k = arr(&ttt_sound["scripts"]["entries"]).len() + scripts.len();
            scripts.insert((c, i), k);
        }
    }
    let mut tags = BTreeMap::new();
    // (the first fighter's that has the alias; the copying character's set has none)
    for set in aliases.values() {
        for (b, m) in set.iter().enumerate().filter_map(|(b, i)| Some((b, d.moves.get((*i)?)?))) {
            let tag =
                d.ttt_cancels(m).into_iter().find(|c| c["kind"] == "command" && c["input"] == "tag" && c["cond"] == 0);
            if let (false, Some(dest)) = (tags.contains_key(&b), tag.and_then(|c| n(&c["dest"]))) {
                tags.insert(b, dest);
            }
        }
    }
    let run_in = aliases.values().find_map(|s| s.get(TAG[2]).copied().flatten()).and_then(|i| d.moves.get(i)).cloned();
    let mut sp = Space { d: &d, p: &p, num, added: Vec::new(), made: BTreeMap::new(), scripts, tags, run_in };
    let mut cov = Coverage::default();
    let (mut index, mut roster) = (d.index.clone(), d.ttt_roster.clone());
    let mut rules = read(ttt, "gamedata/rules.json")?;
    let rules3 = read(t3, "gamedata/rules.json")?;
    let mut sets = Vec::new();
    for &(c, body, _) in &FIGHTERS {
        ctx.log(&format!("[ttt] Tekken 3's character {c} as body {body}"));
        sets.push(moveset(&mut sp, c, FIGHTERS.iter().find(|f| f.0 == c).map_or(0, |f| f.2), &ids, &mut cov)?);
    }
    // the Tekken Tag aliases their moves go to that they lack: the opponent's, its motions listed
    let mut lacked = BTreeSet::new();
    for set in &sets {
        let refs = arr(&set["moves"])
            .iter()
            .map(|m| &m["next"]["alias"])
            .chain(arr(&set["cancels"]).iter().map(|c| &c["dest"]));
        lacked.extend(refs.filter_map(n).filter(|&a| a < TTT_ALIASES && set["aliases"][a]["move"].is_null()));
    }
    // (and the entrance's run in, which they play as their own)
    lacked.insert(TAG[2]);
    let list = motions(&d, &aliases, &lacked);
    ctx.log(&format!("[ttt] {} Tekken Tag motions", list.len()));
    let list = json!(list);
    for (&(c, body, id), set) in FIGHTERS.iter().zip(sets) {
        let file = format!("moveset_{id:02}.json");
        write(ctx, &format!("gamedata/movesets/{file}"), &set)?;
        write(ctx, &format!("gamedata/movesets/motions_{id:02}.json"), &json!({"moveset_id": id, "motions": list}))?;
        let name = arr(&d.roster["characters"])
            .iter()
            .find(|e| n(&e["id"]) == Some(c))
            .and_then(|e| e["name"].as_str())
            .unwrap_or("")
            .to_string();
        let mut selections = Vec::new();
        for e in arr(&d.roster["selections"]["entries"]).iter().filter(|e| n(&e["character"]) == Some(c)) {
            let (Some(record), Some(costume), Some(glb)) = (n(&e["model_record"]), n(&e["costume"]), e["glb"].as_str())
            else {
                continue;
            };
            let bytes = t3.read(&format!("characters/{glb}"))?;
            ctx.out.write(
                &format!("characters/{glb}"),
                &retarget_glb(&bytes).with_context(|| format!("characters/{glb}"))?,
            )?;
            let selection = body * 8 + costume;
            selections.push(selection);
            push(
                &mut index["models"],
                json!({"model": record, "model_record": record, "glb": glb, "body": body, "moveset_id": id, "moveset_file": file}),
            )?;
            push(
                &mut roster["selections"]["entries"],
                json!({"selection": selection, "character": body, "costume": costume, "name": name, "model_record": record, "glb": glb, "moveset_file": format!("movesets/{file}"), "moveset_id": id}),
            )?;
        }
        push(&mut index["fighters"], json!({"name": name, "body": body, "moveset_id": id, "moveset_file": file}))?;
        push(&mut index["files"], json!(file))?;
        push(&mut roster["characters"], json!({"id": body, "name": name, "moveset_id": id, "selections": selections}))?;
        if FIGHTERS[..SELECTABLE].iter().any(|f| f.0 == c) {
            let slot = arr(&roster["select_grid"]["entries"]).len();
            push(&mut roster["select_grid"]["entries"], json!({"slot": slot, "character": body, "name": name}))?;
            // the select's 3D model (Tekken Tag's aliases 0 and 3; Tekken 3's select has none): its stance, on a pick
            // the win screen's first win pose (rules.json `win_lose_motions`); one lying (Doctor B.'s) a round start
            // instead; lifted to Tekken Tag's height
            let models = &mut roster["front_end"]["select"]["models"]["motions"];
            if models.is_array() {
                let t3 = d.t3_set(c).context("no moveset")?;
                let standing = |a: usize, or: usize| -> Result<&str> {
                    let mv = |a: usize| d.t3_move(t3, a).with_context(|| format!("no move for alias {a}"));
                    let m = mv(a)?;
                    let m = if arr(&m["posture"]["names"]).contains(&json!("lying")) { mv(or)? } else { m };
                    m["anim"].as_str().context("no motion")
                };
                self::set(models, body, json!([standing(3, 0)?, standing(3431, 2)?, SELECT_TOP]))?;
            }
        }
        // the hurt and push spheres' radii (both games' layouts are the same); the health (Tekken 3 has one for all)
        for k in ["hurt_spheres", "push_spheres"] {
            let radii = arr(&rules3[k]["per_character"])
                .iter()
                .find(|e| n(&e["character"]) == Some(c))
                .map(|e| e["radii"].clone())
                .context("no radii")?;
            push(&mut rules[k]["per_character"], json!({"character": body, "name": name, "radii": radii}))?;
        }
        let pct = &mut rules["health"]["percent_per_character"];
        while arr(pct).len() <= body {
            push(pct, json!(100))?;
        }
    }
    let mut reactions = d.ttt_reactions.clone();
    for e in std::mem::take(&mut sp.added) {
        push(&mut reactions["entries"], e)?;
    }
    write(ctx, "gamedata/movesets/index.json", &index)?;
    let named = [FIGHTERS[0], FIGHTERS[1]].map(|(c, body, _)| {
        (
            body,
            c,
            arr(&d.roster["characters"])
                .iter()
                .find(|e| n(&e["id"]) == Some(c))
                .and_then(|e| e["name"].as_str())
                .unwrap_or(""),
        )
    });
    front_end(ctx, t3, &mut roster, named)?;
    write(ctx, "gamedata/roster.json", &roster)?;
    write(ctx, "gamedata/rules.json", &rules)?;
    write(ctx, "gamedata/reactions.json", &reactions)?;
    let (mut sound, mut audio) = voices(ctx, t3, &d, ttt_sound, read(ttt, "audio/index.json")?)?;
    self::scripts(ctx, t3, &d, &sp.scripts, &mut sound, &mut audio)?;
    ball(ctx, t3, &sp, &ids, &mut sound, &mut audio)?;
    force(ctx, t3, &d, &ids, &mut sound, &mut audio)?;
    hud(ctx, t3)?;
    effects(ctx, t3, &d)?;
    write(ctx, "gamedata/sound.json", &sound)?;
    write(ctx, "audio/index.json", &audio)?;
    write(ctx, "gamedata/ai.json", &ai(&sp, &read(t3, "gamedata/ai.json")?, read(ttt, "gamedata/ai.json")?)?)?;
    ctx.log(&format!("[ttt] {cov:?}"));
    Ok(cov)
}

/// Tekken Tag's AI data (`ai`) with Tekken 3's for the crossover's fighters, as far as its AI port reads it: their
/// spacing rows (by body), the handler a CPU takes against them (its aliases in Tekken Tag's numbering), and the
/// rules naming their move sets (Gon's low fire to dodge, never thrown; Doctor B.'s wake-up forward).
fn ai(sp: &Space, ai3: &Value, mut ai: Value) -> Result<Value> {
    let (ms, _) = ids(sp.d);
    let to_ms = |m: &Value| m.as_u64().and_then(|m| ms.get(&(m as usize))).copied();
    for &(c, b, id) in &FIGHTERS {
        if let Some(e) = arr(&ai3["spacing"]["entries"]).iter().find(|e| n(&e["character"]) == Some(c)) {
            push(&mut ai["spacing"]["entries"], json!({"character": b, "name": e["name"], "values": e["values"]}))?;
        }
        let t3_id = arr(&sp.d.roster["selections"]["entries"])
            .iter()
            .find(|e| n(&e["character"]) == Some(c))
            .and_then(|e| n(&e["moveset_id"]));
        let Some(h) = arr(&ai3["handlers"]["entries"]).iter().find(|h| n(&h["moveset"]) == t3_id) else { continue };
        for k in ["vs_handler", "self_handler"] {
            let Some(addr) = h[k].as_str() else { continue };
            let mut r = ai3["handlers"]["routines"][addr].clone();
            for list in r["aliases"].as_object_mut().into_iter().flat_map(|m| m.values_mut()) {
                *list = Value::Array(arr(list).iter().filter_map(n).map(|a| json!(sp.alias(a))).collect());
            }
            ai["handlers"]["routines"][addr] = r;
        }
        push(
            &mut ai["handlers"]["entries"],
            json!({"moveset": id, "vs_handler": h["vs_handler"], "self_handler": h["self_handler"]}),
        )?;
    }
    let (r3, r) = (&ai3["rules"], &mut ai["rules"]);
    let ours = |m: Option<usize>| m.filter(|m| FIGHTERS.iter().any(|f| f.2 == *m));
    for k in ["low_fire", "fire", "unthrown"] {
        if let Some(m) = ours(to_ms(&r3[k]["moveset"])) {
            r[k]["moveset"] = json!(m);
            if r3[k].get("descriptor").is_some() {
                r[k]["descriptor"] = r3[k]["descriptor"].clone();
            }
        }
    }
    for m in arr(&r3["wake_forward"]["movesets"]).iter().filter_map(|m| ours(to_ms(m))) {
        push(&mut r["wake_forward"]["movesets"], json!(m))?;
    }
    Ok(ai)
}

// ---------------------------------------------------------------------------------------------------------------
// The sounds

/// The first sound id of the crossover's voice groups: past Tekken Tag's (to 800) and its arena ids (0x321-0x326).
const FIRST_VOICE_ID: usize = 0x330;

/// A Tekken 3 move's sound entries in Tekken Tag's: the voices (types 2, 3: the voice row's categories 0, 1 in both
/// games) as they are, the hit sounds (1, 7) Tekken Tag's generic impact (0x7000: by damage and limb), a timed script
/// (8) its Tekken Tag index (`script`, [`scripts`]), the jump's mute (9, entry 0: Tekken Tag's 0x376578 too); Tekken
/// 3's other table sounds dropped (their numbers are not Tekken Tag's).
fn move_sounds(sounds: &Value, script: impl Fn(usize) -> Option<usize>) -> Value {
    let list: Vec<Value> = arr(sounds)
        .iter()
        .filter_map(|e| match e["type"].as_u64()? {
            t @ (2 | 3) => Some(json!({"type": t, "index": e["index"]})),
            1 | 7 => Some(json!({"type": 7, "index": 0})),
            8 => Some(json!({"type": 8, "index": script(n(&e["index"])?)?})),
            9 if e["index"] == 0 => Some(json!({"type": 9, "index": 0})),
            _ => None,
        })
        .collect();
    if list.is_empty() { Value::Null } else { Value::Array(list) }
}

/// The sample a Tekken 3 SePlay code plays (`own`: the character's bank), as the game's SePlay finds it.
fn sample<'a>(samples: &[&'a Value], code: u64, own: &str) -> Result<&'a Value> {
    // (SePlay: bits 15-14 the bank, 5-4 the program, 3-0 the tone; a bank without tone indices by key range)
    let key = match code >> 14 {
        1 => "sys",
        2 => "ann",
        _ => own,
    };
    let (program, tone) = ((code >> 4) & 3, code & 0xF);
    let tones = |s: &Value| -> Vec<[u64; 4]> {
        arr(&s["tones"]).iter().map(|t| std::array::from_fn(|k| t[k].as_u64().unwrap_or(u64::MAX))).collect()
    };
    let in_bank = || samples.iter().filter(|s| s["file"].as_str().is_some_and(|f| f.starts_with(&format!("{key}/"))));
    in_bank()
        .find(|s| tones(s).iter().any(|t| t[0] == program && t[3] == tone))
        .or_else(|| {
            in_bank().find(|s| {
                tones(s).iter().any(|t| t[0] == program && t[3] == u64::MAX && (t[1]..=t[2]).contains(&(tone + 60)))
            })
        })
        .copied()
        .with_context(|| format!("{own}: no sample for sound code {code:#x}"))
}

/// The fighters' voices as Tekken Tag's: each one's Tekken 3 voice row (the selection's `player_1A`) a new voice row,
/// its samples (Tekken 3's codes resolved in the character's own bank, as the game's SePlay does) a new bank under
/// audio/sfx/ and a new voice group of ids from [`FIRST_VOICE_ID`], each costume's file on that bank and row. Returns
/// sound.json and audio/index.json with them added.
fn voices(ctx: &Ctx, t3: &dyn Vfs, d: &Data, mut sound: Value, mut audio: Value) -> Result<(Value, Value)> {
    let (s3, a3) = (read(t3, "gamedata/sound.json")?, read(t3, "audio/index.json")?);
    let mut bank = arr(&audio["sfx"]).iter().filter_map(|b| n(&b["bank"])).max().unwrap_or(0) + 1;
    let mut id = FIRST_VOICE_ID;
    for &(c, _, _) in &FIGHTERS {
        let sels: Vec<&Value> = arr(&d.roster["selections"]["entries"])
            .iter()
            .filter(|e| n(&e["character"]) == Some(c) && e["model_record"].is_u64())
            .collect();
        let first = sels.first().with_context(|| format!("Tekken 3's character {c}: no model"))?;
        let row = &s3["voice_rows"]["rows"][n(&first["descriptor"]["player_1A"]).context("no voice row")?];
        let own = format!("m{:03}", n(&first["model_record"]).unwrap_or(0));
        let samples: Vec<&Value> = arr(&a3["sfx"]).iter().flat_map(|b| arr(&b["samples"])).collect();
        let mut files = Vec::new();
        for code in arr(&row["codes"]).iter().filter_map(|c| c.as_u64()) {
            files.push(sample(&samples, code, &own)?);
        }
        // (Crow's soldiers have no voice)
        if files.is_empty() {
            continue;
        }
        let mut list = Vec::new();
        let mut group = Vec::new();
        for (k, s) in files.iter().enumerate() {
            let file = s["file"].as_str().unwrap_or_default();
            ctx.out.write(&format!("audio/sfx/{bank:03}/{k:03}.wav"), &t3.read(&format!("audio/sfx/{file}"))?)?;
            let rate = s["rate"].as_u64().unwrap_or(22050);
            list.push(
                json!({"file": format!("{bank:03}/{k:03}.wav"), "rate": rate, "seconds": s["seconds"], "name": file}),
            );
            // (played at the WAV's own rate: pitch x 48000 / 4096 = rate)
            group.push(json!({"pitch": (rate * 4096 + 24000) / 48000, "volume": 25, "channel": 6}));
        }
        let (r, count) = (arr(&sound["voice_rows"]["rows"]).len(), files.len());
        let g = arr(&sound["groups"]["entries"]).len();
        push(
            &mut sound["groups"]["entries"],
            json!({"group": g, "first_id": id, "count": count, "kind": 2, "bank": null, "voice_row": r, "samples": group}),
        )?;
        push(
            &mut sound["voice_rows"]["rows"],
            json!({"row": r, "counts": row["counts"], "codes": (id..id + count).collect::<Vec<_>>(), "cooldown": 10}),
        )?;
        for e in &sels {
            push(
                &mut sound["voice_banks"]["characters"],
                json!({"model_record": e["model_record"], "bank": bank, "voice_row": r}),
            )?;
        }
        push(&mut audio["sfx"], json!({"bank": bank, "samples": list}))?;
        (bank, id) = (bank + 1, id + count);
    }
    Ok((sound, audio))
}

/// The timed sound scripts (move sound entries of type 8) of Tekken 3's `moves`, each once, in order.
fn script_ids<'a>(moves: impl Iterator<Item = &'a Value>) -> BTreeSet<usize> {
    moves
        .flat_map(|m| arr(&m["sounds"]))
        .filter(|e| e["type"] == 8 && e["index"] != 255)
        .filter_map(|e| n(&e["index"]))
        .collect()
}

/// The fighters' Tekken 3 sound scripts (their falls, throws, wake-ups, win poses: 0x800971F8, [cmd, frame, code])
/// as Tekken Tag's (0x505C10, [frame, kind, entry]) at the indices `ids` gave them, appended to `sound`'s: the voices
/// (entry types 2-5) as they are, a table sound (cmd 0, the other types: sfx_table, then the SePlay code resolved in
/// the fighter's bank, as the game's SePlay does) a new id of a group of the fighter's (an opponent's own-bank sound,
/// which Tekken Tag's opponent has not, dropped).
fn scripts(
    ctx: &Ctx,
    t3: &dyn Vfs,
    d: &Data,
    ids: &BTreeMap<(usize, usize), usize>,
    sound: &mut Value,
    audio: &mut Value,
) -> Result<()> {
    let (s3, a3) = (read(t3, "gamedata/sound.json")?, read(t3, "audio/index.json")?);
    let (lists, table) = (arr(&s3["scripts"]["entries"]), arr(&s3["sfx_table"]["codes"]));
    let samples: Vec<&Value> = arr(&a3["sfx"]).iter().flat_map(|b| arr(&b["samples"])).collect();
    for &(c, _, _) in &FIGHTERS {
        // (every fighter's lists pushed, in `ids`' order)
        let own = arr(&d.roster["selections"]["entries"])
            .iter()
            .find(|e| n(&e["character"]) == Some(c))
            .and_then(|e| n(&e["model_record"]))
            .map_or("sys".into(), |r| format!("m{r:03}"));
        // each entry: (frame, kind, a voice entry or a SePlay code, which)
        let entries: Vec<Vec<(u64, u64, u64, bool)>> = ids
            .range((c, 0)..=(c, usize::MAX))
            .map(|(&(_, i), _)| {
                arr(lists.get(i).unwrap_or(&Value::Null))
                    .iter()
                    .filter_map(|e| {
                        let [cmd, frame, code] = [0, 1, 2].map(|k| e[k].as_u64());
                        let (cmd, frame, code) = (cmd?, frame?, code?);
                        if cmd != 0 && (2..=5).contains(&(code >> 12)) {
                            return Some((frame, cmd, code, false));
                        }
                        let seplay = table.get((if cmd == 0 { code } else { code & 0xFFF }) as usize)?.as_u64()?;
                        let there = seplay != 0 && !(cmd == 2 && seplay >> 14 == 3);
                        there.then_some((frame, cmd, seplay, true))
                    })
                    .collect()
            })
            .collect();
        let mut codes = Vec::new();
        for &(_, _, code, _) in entries.iter().flatten().filter(|e| e.3) {
            if !codes.contains(&code) && sample(&samples, code, &own).is_ok() {
                codes.push(code);
            }
        }
        let first = if codes.is_empty() { 0 } else { sound_group(ctx, t3, &codes, &own, sound, audio)? };
        for list in entries {
            let list: Vec<Value> = list
                .into_iter()
                .filter_map(|(frame, cmd, code, seplay)| {
                    if !seplay {
                        return Some(json!([frame, cmd, code]));
                    }
                    let id = (first + codes.iter().position(|&x| x == code)?) as u64;
                    Some(json!([frame, cmd, if cmd == 0 { id } else { 1 << 12 | id }]))
                })
                .collect();
            push(&mut sound["scripts"]["entries"], Value::Array(list))?;
        }
    }
    Ok(())
}

/// The back steps Ball pushes back, Tekken 3's aliases to Tekken Tag's (the same moves: a back dash, then the back
/// walk or the stance; the pairing leaves them, their frames differ).
const BACK_STEPS: [(usize, usize); 4] = [(30, 70), (31, 71), (32, 72), (33, 73)];
/// Tekken 3's Ball court in Tekken Tag: its arena id, past Tekken Tag's 23 (0-22).
pub const BALL_STAGE: usize = 23;

/// Tekken 3's Ball under Tekken Tag (OpenKumite's mode there): Tekken 3's ball.json with its numbers Tekken Tag's,
/// written with the court (Tekken 3's beach, as arena [`BALL_STAGE`]) and the Ball sounds (Tekken 3's samples: a new
/// bank and a group of new ids in `sound` / `audio`). Renumbered: the moves pushed back (aliases, as the pairing has
/// them), the self-charging move set and its exception, the CPU's pool (bodies; a character Tekken Tag lacks is
/// left out). A move's ball power needs no table: Tekken 3 takes it from the move's damage and hit mask (0x80047D98),
/// which Tekken Tag's moves carry in the same form. Nothing without Tekken 3's ball.json.
fn ball(ctx: &Ctx, t3: &dyn Vfs, sp: &Space, ids: &Ids, sound: &mut Value, audio: &mut Value) -> Result<()> {
    if !t3.exists("gamedata/ball.json") {
        return Ok(());
    }
    let mut b = read(t3, "gamedata/ball.json")?;
    let c = &mut b["constants"];
    // (an alias in Gon's and Doctor B.'s numbering, else the pairing's)
    let alias = |a: &Value| n(a).and_then(|a| sp.num.get(&a).or(sp.p.alias.get(&a)).copied());
    // each pushed alias in both: Gon's and Doctor B.'s, and Tekken Tag's fighters' (the pairing's, else BACK_STEPS)
    for p in c["moves"]["push_back"].as_array_mut().context("ball.json: push_back")? {
        let mut to: BTreeSet<usize> = BTreeSet::new();
        for a in arr(&p["aliases"]).iter().filter_map(n) {
            to.extend(sp.num.get(&a));
            to.extend(sp.p.alias.get(&a).or(BACK_STEPS.iter().find(|s| s.0 == a).map(|s| &s.1)));
        }
        p["aliases"] = json!(to);
    }
    let charge = &mut c["moves"]["self_charge"];
    let (set, except) = (n(&charge["moveset"]).and_then(|m| ids.0.get(&m)).copied(), alias(&charge["except_alias"]));
    (charge["moveset"], charge["except_alias"]) = (json!(set), json!(except));
    let pool: Vec<usize> = arr(&c["cpu"]["pool"]).iter().filter_map(|k| ids.1.get(&n(k)?).copied()).collect();
    c["cpu"]["pool"] = json!(pool);
    c["cpu"]["first_ever"] = json!(n(&c["cpu"]["first_ever"]).and_then(|k| ids.1.get(&k)));
    let court = n(&c["stage"]).context("ball.json: no stage")?;
    c["stage"] = json!(BALL_STAGE);
    ctx.out.write(&format!("stages/stage_{BALL_STAGE:02}.glb"), &t3.read(&format!("stages/stage_{court:02}.glb"))?)?;
    // (its light: the ball's shadow takes the court's shadow colour)
    add_stages(ctx, t3, &[(court, BALL_STAGE)])?;
    // the sounds: each code once, as new ids
    let codes = codes_in(c["sounds"].as_object().context("ball.json: sounds")?.values());
    let first = sound_group(ctx, t3, &codes, "sys", sound, audio)?;
    c["sounds"].as_object_mut().into_iter().flat_map(|m| m.values_mut()).for_each(|v| renumber(v, &codes, first));
    ctx.log(&format!(
        "[ttt] Ball: {}, self charge {set:?} / {except:?}, {} sounds",
        b["constants"]["moves"]["push_back"],
        codes.len()
    ));
    write(ctx, "gamedata/ball.json", &b)?;
    // Tekken 3's fight camera, Ball's (it frames the ball too, 0x80064E10): the moves' yaw caps in Tekken Tag's camera
    // ids (Tekken 3's 0x27-0x2A are Tekken Tag's 0x13-0x16, as the moves' own)
    let mut cam = read(t3, "gamedata/camera.json")?;
    if let Some(caps) = cam["fight"]["yaw"]["move_caps"].as_object_mut() {
        let id = |k: &str| u64::from_str_radix(k.trim_start_matches("0x"), 16).ok()?.checked_sub(0x14);
        *caps = std::mem::take(caps).into_iter().filter_map(|(k, v)| Some((format!("0x{:X}", id(&k)?), v))).collect();
    }
    write(ctx, "gamedata/ball_camera.json", &cam)
}

/// Tekken Force's stages in Tekken Tag: arena ids past Ball's court (Tekken 3's 15-18 as 24-27).
pub const FORCE_STAGE: usize = 24;
/// force.json's sounds (SePlay codes): the chicken's, the countdown voices, the tally's.
const FORCE_SOUNDS: [&str; 5] = ["/chicken/sfx", "/chicken/voice", "/countdown/codes", "/tally/sfx", "/tally/key_sfx"];

/// Tekken 3's Tekken Force under Tekken Tag (OpenKumite's mode there): Tekken 3's force.json with its numbers Tekken
/// Tag's, written with its stages (as arenas from [`FORCE_STAGE`]; their lights, which the blob shadows' colour is
/// part of, added to stages.json), the chickens' pictures, the final stage's music (a new BGM id) and its
/// sounds (a group of new ids in `sound` / `audio`). Renumbered: the soldiers' selection (Crow's: a crossover
/// fighter), the bosses and the final character (Tekken Tag's bodies of the same name), the boss routes by Tekken
/// Tag's selection (body x 8 + costume; a fighter without a Tekken 3 namesake takes the route of Tekken 3's
/// character body % 14). Nothing without Tekken 3's force.json.
fn force(ctx: &Ctx, t3: &dyn Vfs, d: &Data, ids: &Ids, sound: &mut Value, audio: &mut Value) -> Result<()> {
    if !t3.exists("gamedata/force.json") {
        return Ok(());
    }
    let mut f = read(t3, "gamedata/force.json")?;
    let body = |c: Option<usize>| c.and_then(|c| ids.1.get(&c)).copied();
    // the stages
    let mut olds: Vec<usize> = arr(&f["stages"]).iter().filter_map(|s| n(&s["stage"])).collect();
    olds.sort_unstable();
    olds.dedup();
    let to = |s: Option<usize>| s.and_then(|s| olds.iter().position(|&o| o == s)).map(|k| FORCE_STAGE + k);
    for (k, &old) in olds.iter().enumerate() {
        let id = FORCE_STAGE + k;
        ctx.out.write(&format!("stages/stage_{id:02}.glb"), &t3.read(&format!("stages/stage_{old:02}.glb"))?)?;
    }
    add_stages(ctx, t3, &olds.iter().enumerate().map(|(k, &old)| (old, FORCE_STAGE + k)).collect::<Vec<_>>())?;
    for s in f["stages"].as_array_mut().context("force.json: stages")? {
        s["stage"] = json!(to(n(&s["stage"])));
    }
    let slanted = &mut f["constants"]["bounds"]["slanted_stage"];
    *slanted = json!(to(n(slanted)));
    // the soldiers, the bosses
    let base = n(&f["soldier_selection_base"]);
    let e = arr(&d.roster["selections"]["entries"])
        .iter()
        .find(|e| n(&e["selection"]) == base)
        .context("force.json: no soldier selection")?;
    let soldier = body(n(&e["character"])).context("the soldiers: not installed")? * 8 + n(&e["costume"]).unwrap_or(0);
    f["soldier_selection_base"] = json!(soldier);
    let t3_of: BTreeMap<usize, usize> = ids.1.iter().map(|(&c, &b)| (b, c)).collect();
    let routes = &f["boss_routes"];
    let route = |r: &Value| -> Value { arr(r).iter().map(|b| json!([body(n(&b[0])).unwrap_or(0), b[1]])).collect() };
    let bodies = t3_of.keys().max().map_or(0, |b| b + 1);
    let by_selection: Vec<Value> = (0..bodies * 8)
        .map(|s| {
            let c = t3_of.get(&(s / 8)).copied().unwrap_or(s / 8 % SHARED);
            route(&routes["by_selection"][c * 4 + s % 4])
        })
        .collect();
    let otherwise = json!([body(n(&routes["default"][0])).unwrap_or(0), routes["default"][1]]);
    (f["boss_routes"]["by_selection"], f["boss_routes"]["default"]) = (json!(by_selection), otherwise);
    let last = &mut f["constants"]["final"]["character"];
    *last = json!(body(n(last)).context("force.json: the final character is not installed")?);
    // the final stage's music: Tekken 3's track as a new BGM
    let mut music = read(ctx.out, "gamedata/music.json")?;
    let bgm = n(&f["constants"]["final"]["bgm"]).context("force.json: no final bgm")?;
    let col = read(t3, "gamedata/music.json")?["bgm"][bgm]["columns"][0].clone();
    let file = col["file"].as_str().context("music.json: the final bgm has no file")?;
    ctx.out.write(&format!("audio/music/t3_{file}"), &t3.read(&format!("audio/music/{file}"))?)?;
    let id = arr(&music["bgm"]).len();
    push(
        &mut music["bgm"],
        json!({"id": id, "columns": [{"kind": 1, "file": format!("t3_{file}"), "loop": col["loop"]}], "characters": []}),
    )?;
    f["constants"]["final"]["bgm"] = json!(id);
    write(ctx, "gamedata/music.json", &music)?;
    // the sounds, the chickens' pictures
    let c = &mut f["constants"];
    let codes = codes_in(FORCE_SOUNDS.iter().filter_map(|p| c.pointer(p)));
    let first = sound_group(ctx, t3, &codes, "sys", sound, audio)?;
    for p in FORCE_SOUNDS {
        if let Some(v) = c.pointer_mut(p) {
            renumber(v, &codes, first);
        }
    }
    for e in t3.list("pictures/force")?.into_iter().filter(|e| !e.is_dir) {
        ctx.out.write(&format!("pictures/force/{}", e.name), &t3.read(&format!("pictures/force/{}", e.name))?)?;
    }
    ctx.log(&format!(
        "[ttt] Force: stages {olds:?} from {FORCE_STAGE}, soldiers {soldier}, final bgm {id}, {} sounds",
        codes.len()
    ));
    write(ctx, "gamedata/force.json", &f)
}

/// Gon's fire and gas as Tekken 3 draws them: its effects atlas beside Tekken Tag's (`t3_atlas.png`) and, in Tekken
/// Tag's effects.json, `t3`: that sheet and Tekken 3's strips of them. The fighters' own sparks (Doctor B.'s blue
/// lightning, ...: Tekken Tag's effects.json has none of them, so their heavy hits and move ops 6 / 7 showed nothing)
/// by model record in its `characters`, as Tekken 3 draws them (a strip of that sheet, no burst), with their light.
/// Nothing when either game has no effects.
fn effects(ctx: &Ctx, t3: &dyn Vfs, d: &Data) -> Result<()> {
    const DIR: &str = "pictures/effects";
    let (Ok(mut fx), Ok(fx3)) =
        (read(ctx.out, &format!("{DIR}/effects.json")), read(t3, &format!("{DIR}/effects.json")))
    else {
        return Ok(());
    };
    let strips: serde_json::Map<String, Value> = ["fire", "fire_end", "gas", "mist"]
        .iter()
        .filter_map(|k| Some((k.to_string(), fx3["strips"].get(*k)?.clone())))
        .collect();
    ctx.out.write(&format!("{DIR}/t3_atlas.png"), &t3.read(&format!("{DIR}/atlas.png"))?)?;
    let a = &fx3["atlas"];
    let sheet = json!({"file": "t3_atlas.png", "cols": a["cols"], "rows": a["rows"]});
    for e in arr(&d.roster["selections"]["entries"])
        .iter()
        .filter(|e| FIGHTERS.iter().any(|f| n(&e["character"]) == Some(f.0)))
    {
        let Some(record) = n(&e["model_record"]).map(|r| r.to_string()) else { continue };
        if let Some(c) = fx3["characters"].get(&record) {
            fx["characters"][&record] =
                json!({"cells": c["cells"], "life": c["life"], "light": c["light"], "sheet": sheet});
        }
    }
    if !strips.is_empty() {
        fx["t3"] = json!({"sheet": sheet, "strips": strips});
    }
    write(ctx, &format!("{DIR}/effects.json"), &fx)
}

/// Where Tekken 3's HUD goes in Tekken Tag's extras ([`hud`]), and the picture sets it draws from: the in-game
/// HUD and its fonts (the executable's), Tekken Force's overlay (its keys).
pub const T3_HUD: &str = "t3";
const HUD_SETS: [&str; 2] = ["exe", "r007_raw"];

/// Tekken 3's HUD for its modes under Tekken Tag (Tekken Force's bars, name plates, keys, progress map and results;
/// Tekken Ball's gauges): its pictures of [`HUD_SETS`] with their index and palettes, and screens.json, under
/// [`T3_HUD`]/ as in Tekken 3's root. Nothing without Tekken 3's Force or Ball.
fn hud(ctx: &Ctx, t3: &dyn Vfs) -> Result<()> {
    if !t3.exists("gamedata/force.json") && !t3.exists("gamedata/ball.json") {
        return Ok(());
    }
    let ours = |file: &Value| file.as_str().and_then(|f| f.split('/').next()).is_some_and(|s| HUD_SETS.contains(&s));
    let mut index = read(t3, "pictures/index.json")?;
    let sets = index.as_array_mut().context("pictures/index.json: not a list")?;
    for s in sets.iter_mut() {
        if let Some(images) = s["images"].as_array_mut() {
            images.retain(|i| ours(&i["file"]));
        }
    }
    sets.retain(|s| !arr(&s["images"]).is_empty());
    write(ctx, &format!("{T3_HUD}/pictures/index.json"), &index)?;
    let mut cluts = read(t3, "pictures/cluts.json")?;
    if let Some(c) = cluts["cluts"].as_object_mut() {
        c.retain(|source, _| HUD_SETS.contains(&source.as_str()));
    }
    write(ctx, &format!("{T3_HUD}/pictures/cluts.json"), &cluts)?;
    for set in HUD_SETS {
        for e in t3.list(&format!("pictures/{set}"))?.into_iter().filter(|e| !e.is_dir) {
            let file = format!("pictures/{set}/{}", e.name);
            ctx.out.write(&format!("{T3_HUD}/{file}"), &t3.read(&file)?)?;
        }
    }
    ctx.out.write(&format!("{T3_HUD}/gamedata/screens.json"), &t3.read("gamedata/screens.json")?)
}

/// Tekken 3's stages `(old, new)` added to Tekken Tag's stages.json as arenas `new`: their name and light table (the
/// fighters' light, the shadows' colour), beside Tekken Tag's own entries.
fn add_stages(ctx: &Ctx, t3: &dyn Vfs, pairs: &[(usize, usize)]) -> Result<()> {
    let stages3 = read(t3, "gamedata/stages.json")?;
    let path = "gamedata/stages.json";
    let mut stages = if ctx.out.exists(path) { read(ctx.out, path)? } else { json!({"stages": []}) };
    for &(old, id) in pairs {
        if let Some(e) = arr(&stages3["stages"]).iter().find(|e| n(&e["id"]) == Some(old)) {
            push(&mut stages["stages"], json!({"id": id, "name": e["name"], "light": e["light"]}))?;
        }
    }
    write(ctx, path, &stages)
}

/// Each SePlay code in `vals` (numbers and lists of them) once, in order.
fn codes_in<'a>(vals: impl Iterator<Item = &'a Value>) -> Vec<u64> {
    let mut codes = Vec::new();
    for v in vals.flat_map(|v| v.as_array().map_or_else(|| vec![v], |l| l.iter().collect())) {
        if let Some(code) = v.as_u64().filter(|c| !codes.contains(c)) {
            codes.push(code);
        }
    }
    codes
}

/// The codes in `v` (a number or a list of them) as their new ids: `first` + their place in `codes`.
fn renumber(v: &mut Value, codes: &[u64], first: usize) {
    let one = |x: &mut Value| {
        if let Some(k) = x.as_u64().and_then(|code| codes.iter().position(|&c| c == code)) {
            *x = json!(first + k);
        }
    };
    match v.as_array_mut() {
        Some(list) => list.iter_mut().for_each(one),
        None => one(v),
    }
}

/// Tekken 3's sounds `codes` (SePlay codes; `own`: the character bank's) as a group of new Tekken Tag ids, from past the last group's, on
/// a bank of their samples: added to `sound` / `audio`. Returns the first id (code k plays as first + k).
fn sound_group(
    ctx: &Ctx,
    t3: &dyn Vfs,
    codes: &[u64],
    own: &str,
    sound: &mut Value,
    audio: &mut Value,
) -> Result<usize> {
    let a3 = read(t3, "audio/index.json")?;
    let samples: Vec<&Value> = arr(&a3["sfx"]).iter().flat_map(|b| arr(&b["samples"])).collect();
    let bank = arr(&audio["sfx"]).iter().filter_map(|b| n(&b["bank"])).max().unwrap_or(0) + 1;
    let groups = arr(&sound["groups"]["entries"]);
    let first = groups.iter().filter_map(|g| Some(n(&g["first_id"])? + n(&g["count"])?)).max().unwrap_or(0);
    let g = groups.len();
    let (mut list, mut group) = (Vec::new(), Vec::new());
    for (k, &code) in codes.iter().enumerate() {
        let s = sample(&samples, code, own)?;
        let file = s["file"].as_str().unwrap_or_default();
        ctx.out.write(&format!("audio/sfx/{bank:03}/{k:03}.wav"), &t3.read(&format!("audio/sfx/{file}"))?)?;
        let rate = s["rate"].as_u64().unwrap_or(22050);
        list.push(
            json!({"file": format!("{bank:03}/{k:03}.wav"), "rate": rate, "seconds": s["seconds"], "name": file}),
        );
        group.push(json!({"pitch": (rate * 4096 + 24000) / 48000, "volume": 25, "channel": 6}));
    }
    push(
        &mut sound["groups"]["entries"],
        json!({"group": g, "first_id": first, "count": codes.len(), "kind": 0, "bank": bank, "voice_row": null, "samples": group}),
    )?;
    push(&mut audio["sfx"], json!({"bank": bank, "samples": list}))?;
    Ok(first)
}

// ---------------------------------------------------------------------------------------------------------------
// The front end: faces, portraits, name plates, the select grid's slots

fn png(fs: &dyn Vfs, path: &str) -> Result<RgbaImage> {
    Ok(image::load_from_memory_with_format(
        &fs.read(path).with_context(|| format!("reading {path}"))?,
        image::ImageFormat::Png,
    )
    .with_context(|| format!("decoding {path}"))?
    .to_rgba8())
}

/// Tekken 3's portrait of a character (126 x 252) as Tekken Tag's pictures: the select face (52 x 52: the face,
/// darker and warmer as Tekken Tag's are, on a dark ground in the frame of Tekken Tag's face `frame`) and the portrait (188 x 256:
/// the top, to the width).
fn faces(portrait: &RgbaImage, frame: &RgbaImage) -> (RgbaImage, RgbaImage) {
    use image::imageops::{FilterType, crop_imm, resize};
    let head = resize(&crop_imm(portrait, 16, 40, 94, 94).to_image(), 48, 48, FilterType::Triangle);
    let mut face = frame.clone();
    for (x, y, p) in head.enumerate_pixels() {
        let tone = |c: u8, k: u32| (c as u32 * k / 100).min(255) as u8;
        let a = p[3] as u32;
        let bg = [40, 24, 20];
        let mix = |c: u8, b: u8| ((c as u32 * a + b as u32 * (255 - a)) / 255) as u8;
        face.put_pixel(
            x + 2,
            y + 2,
            Rgba([mix(tone(p[0], 85), bg[0]), mix(tone(p[1], 75), bg[1]), mix(tone(p[2], 65), bg[2]), 255]),
        );
    }
    let big = resize(portrait, 188, 376, FilterType::Triangle);
    (face, crop_imm(&big, 0, 0, 188, 256).to_image())
}

/// A name plate (128 x 14) in Tekken Tag's big font (`font`: 52 x 42 glyphs in `rows`), right-aligned as its
/// plates are.
fn plate(font: &RgbaImage, rows: &[&str], name: &str) -> RgbaImage {
    use image::imageops::{FilterType, crop_imm, resize};
    let mut out = RgbaImage::new(128, 14);
    let glyphs: Vec<RgbaImage> = name
        .to_uppercase()
        .chars()
        .filter_map(|ch| {
            if ch == ' ' {
                return Some(RgbaImage::new(17, 14));
            }
            let (r, row) = rows.iter().enumerate().find(|r| r.1.contains(ch))?;
            let col = row.chars().position(|c| c == ch)? as u32;
            Some(resize(&crop_imm(font, col * 52, r as u32 * 42, 52, 42).to_image(), 17, 14, FilterType::Triangle))
        })
        .collect();
    // (the glyphs' cells are wider than their letters: 12 apart, closer for a long name)
    let step = (126 / glyphs.len().max(1) as i64).min(12);
    let mut x = 128 - 17 - step * (glyphs.len() as i64 - 1);
    for g in glyphs {
        for (gx, gy, p) in g.enumerate_pixels() {
            let tx = x + gx as i64;
            if (0..128).contains(&tx) && p[3] > 0 {
                out.put_pixel(tx as u32, gy, *p);
            }
        }
        x += step;
    }
    out
}

/// Adds the select grid's slots on the edges beside its bottom row (next to the two hidden ones there, Mokujin's
/// and Unknown's) for `bodies` (left, right), with their faces, portraits and name plates; selectable from the start.
fn front_end(ctx: &Ctx, t3: &dyn Vfs, roster: &mut Value, bodies: [(usize, usize, &str); 2]) -> Result<()> {
    let ttt = ctx.out;
    let fe = &mut roster["front_end"];
    let slots = fe["select"]["slots"].as_array_mut().context("roster.json: no select slots")?;
    // the bottom row's ends (its hidden slots: the leftmost and rightmost)
    let bottom = slots.iter().filter_map(|s| s["y"].as_i64()).max().context("no slots")?;
    let row: Vec<(i64, usize)> = slots
        .iter()
        .filter(|s| s["y"].as_i64() == Some(bottom))
        .filter_map(|s| Some((s["x"].as_i64()?, n(&s["slot"])?)))
        .collect();
    let (Some(&(lx, left)), Some(&(rx, right))) = (row.iter().min(), row.iter().max()) else { bail!("no bottom row") };
    let dx = 53;
    let above = |x: i64| {
        slots
            .iter()
            .find(|s| s["y"].as_i64() == Some(bottom - 54) && s["x"].as_i64() == Some(x))
            .and_then(|s| n(&s["slot"]))
    };
    let (l, r) = (slots.len(), slots.len() + 1);
    let (up_l, up_r) = (above(lx - dx), above(rx + dx));
    slots.push(json!({"slot": l, "up": up_l, "down": null, "left": r, "right": left, "x": lx - dx, "y": bottom, "character": bodies[0].0}));
    slots.push(json!({"slot": r, "up": up_r, "down": null, "left": right, "right": l, "x": rx + dx, "y": bottom, "character": bodies[1].0}));
    slots[left]["left"] = json!(l);
    slots[right]["right"] = json!(r);
    for (up, s) in [(up_l, l), (up_r, r)] {
        if let Some(u) = up {
            slots[u]["down"] = json!(s);
        }
    }
    let frame = png(ttt, &format!("pictures/{}", fe["select"]["random_face"].as_str().context("no random face")?))?;
    let font = png(ttt, &format!("pictures/{}", fe["fonts"]["big"]["picture"].as_str().context("no big font")?))?;
    let rows: Vec<&str> = arr(&fe["fonts"]["big"]["rows"]).iter().filter_map(|r| r.as_str()).collect();
    let mut made = Vec::new();
    for (body, c, name) in bodies {
        let (face, portrait) = faces(&png(t3, &format!("pictures/portraits/{c:02}.png"))?, &frame);
        let files = [
            format!("crossover_face_{body}.png"),
            format!("crossover_portrait_{body}.png"),
            format!("crossover_plate_{body}.png"),
        ];
        for (f, img) in files.iter().zip([face, portrait, plate(&font, &rows, name)]) {
            crate::export_pictures::save_png(ttt, &format!("pictures/{f}"), &img)?;
        }
        made.push((body, files));
    }
    for (body, [face, portrait, plate]) in made {
        for (list, v) in [("faces", json!(face)), ("portraits", json!(portrait))] {
            set(&mut fe["select"][list], body, v)?;
        }
        set(&mut fe["hud"]["plates"], body, json!({"picture": plate, "x": 30, "streak": 0}))?;
        push(&mut fe["select"]["selectable"], json!(body))?;
    }
    // QUICK SELECT's names (by body x 8 + costume), in capitals as Tekken Tag's
    if fe["select"]["quick"]["names"]["texts"].is_array() {
        for (body, _, name) in bodies {
            for k in 0..8 {
                set(&mut fe["select"]["quick"]["names"]["texts"], body * 8 + k, json!(name.to_uppercase()))?;
            }
        }
    }
    Ok(())
}

/// Entry `k` of a list (the lists are by body: Tekken Tag's own run past its 34, unused).
fn set(list: &mut Value, k: usize, v: Value) -> Result<()> {
    let l = list.as_array_mut().context("not a list")?;
    while l.len() <= k {
        l.push(Value::Null);
    }
    l[k] = v;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ok_converter_common::io::{Layered, StdFs};

    /// Each node's world place at rest.
    fn places(glb: &[u8]) -> Result<Vec<[f64; 3]>> {
        let (j, _) = glb_split(glb)?;
        let nodes = arr(&j["nodes"]);
        let mut world: Vec<Option<(Q, [f64; 3])>> = vec![None; nodes.len()];
        let mut todo: Vec<(usize, Q, [f64; 3])> = (0..nodes.len())
            .filter(|&i| !nodes.iter().any(|p| arr(&p["children"]).iter().any(|c| n(c) == Some(i))))
            .map(|i| (i, [0.0, 0.0, 0.0, 1.0], [0.0; 3]))
            .collect();
        while let Some((i, pq, pt)) = todo.pop() {
            let f = |k: &str, d: usize, z: f64| -> Vec<f64> {
                (0..d).map(|c| nodes[i][k][c].as_f64().unwrap_or(if c == 3 { 1.0 } else { z })).collect()
            };
            let (r, t) = (f("rotation", 4, 0.0), f("translation", 3, 0.0));
            let t = rotate(pq, [t[0], t[1], t[2]]);
            let place = [pt[0] + t[0], pt[1] + t[1], pt[2] + t[2]];
            let q = qmul(pq, [r[0], r[1], r[2], r[3]]);
            world[i] = Some((q, place));
            todo.extend(arr(&nodes[i]["children"]).iter().filter_map(n).map(|c| (c, q, place)));
        }
        Ok(world.into_iter().map(|w| w.map_or([0.0; 3], |w| w.1)).collect())
    }

    /// The jump's mute (type 9 entry 0) is kept; Tekken 3's other type 9 entries and table sounds are dropped.
    #[test]
    fn the_jumps_mute_is_kept() {
        let sounds = json!([{"type": 9, "index": 0}, {"type": 9, "index": 2}, {"type": 5, "index": 1}]);
        assert_eq!(move_sounds(&sounds, |_| None), json!([{"type": 9, "index": 0}]));
    }

    /// A body turned onto Tekken Tag's axes keeps every joint where it was; a hand's axes are a quarter turn off.
    #[test]
    fn turned_bodies_keep_their_joints() -> Result<()> {
        let Ok(root) = std::env::var("OPENKUMITE_ASSETS") else { return Ok(()) };
        let glb = StdFs(root.into()).read("characters/model_239.glb")?;
        let out = retarget_glb(&glb)?;
        for (a, b) in places(&glb)?.iter().zip(places(&out)?) {
            assert!((0..3).all(|k| (a[k] - b[k]).abs() < 1e-9), "{a:?} {b:?}");
        }
        let q = qmul(conj(turn("hand_A")), turn("hand_A"));
        assert!((q[3] - 1.0).abs() < 1e-12);
        Ok(())
    }

    /// On both games' data (OPENKUMITE_ASSETS, OK_TTT_ASSETS): the install writes the same files twice; most of the
    /// fighters' aliases and the reactions they deal find a pair.
    #[test]
    fn installs_the_same_twice() -> Result<()> {
        let (Ok(t3), Ok(ttt)) = (std::env::var("OPENKUMITE_ASSETS"), std::env::var("OK_TTT_ASSETS")) else {
            return Ok(());
        };
        let (t3, ttt) = (StdFs(t3.into()), StdFs(ttt.into()));
        let tmp = std::env::temp_dir().join(format!("ok_t3_ttt_{}", std::process::id()));
        let mut outs = Vec::new();
        for k in 0..2 {
            let top = StdFs(tmp.join(k.to_string()));
            let view = Layered { top: &top, below: vec![&ttt] };
            let cov = install(&Ctx { out: &view, log: &|l| eprintln!("{l}"), jobs: 1 }, &t3)?;
            assert!(
                cov.aliases.iter().all(|&(all, paired)| paired * 3 > all)
                    && cov.dealt.iter().all(|&(all, paired)| paired * 2 > all),
                "{cov:?}"
            );
            // Ball: the back dash pushed in Tekken Tag's alias too, the court, the serve's sound a new group's id
            let ball = read(&top, "gamedata/ball.json")?;
            let c = &ball["constants"];
            assert!(
                arr(&c["moves"]["push_back"][1]["aliases"]).contains(&json!(70)) && c["stage"] == BALL_STAGE,
                "{c}"
            );
            let serve = n(&c["sounds"]["serve"]).context("no serve")?;
            let sound = read(&top, "gamedata/sound.json")?;
            let groups = arr(&sound["groups"]["entries"]);
            assert!(
                groups.iter().any(|g| n(&g["first_id"]) == Some(serve))
                    && top.exists(&format!("stages/stage_{BALL_STAGE:02}.glb"))
            );
            // and Tekken 3's camera, its yaw caps in Tekken Tag's ids
            let cam = read(&top, "gamedata/ball_camera.json")?;
            assert_eq!(cam["fight"]["yaw"]["move_caps"]["0x13"], json!(512), "{}", cam["fight"]["yaw"]);
            // Force: the soldiers Crow's (body 36), its stages from 24, the final character Doctor B., a countdown
            // voice the last group's id
            let f = read(&top, "gamedata/force.json")?;
            let c = &f["constants"];
            assert!(
                f["soldier_selection_base"] == 36 * 8
                    && f["stages"][0]["stage"] == FORCE_STAGE
                    && c["final"]["character"] == 35
                    && top.exists(&format!("stages/stage_{:02}.glb", FORCE_STAGE + 3))
                    && n(&groups.last().context("no groups")?["first_id"]) <= n(&c["countdown"]["codes"][0]),
                "{c}"
            );
            // the select's 3D models: Gon's stance and first win pose, Doctor B.'s round starts; their QUICK SELECT
            // names
            let select = &read(&top, "gamedata/roster.json")?["front_end"]["select"];
            let motions = &select["models"]["motions"];
            assert!(
                motions[34] == json!(["motion_025", "motion_191", SELECT_TOP])
                    && motions[35] == json!(["motion_259", "motion_260", SELECT_TOP]),
                "{motions}"
            );
            // Doctor B.: his backflip's script (Tekken 3's 154: his voice, then the take-off and landing as a group
            // of his) past Tekken Tag's; his spark on Tekken 3's atlas; the tag from more than the stance (lying, his
            // guard), Tekken Tag's run in as the entrance
            let set = read(&top, "gamedata/movesets/moveset_33.json")?;
            let flip = arr(&set["moves"]).iter().find(|m| m["anim"] == "motion_040").context("no backflip")?;
            let script = n(&flip["sounds"][0]["index"]).context("no script")?;
            let base = arr(&read(&ttt, "gamedata/sound.json")?["scripts"]["entries"]).len();
            let list = &read(&top, "gamedata/sound.json")?["scripts"]["entries"][script];
            assert!(script >= base && list[0] == json!([2, 1, 8192]) && list[1][2].as_u64() > Some(0x1000), "{list}");
            let fx = read(&top, "pictures/effects/effects.json")?;
            assert!(
                fx["characters"]["247"]["sheet"]["file"] == "t3_atlas.png"
                    && fx["characters"]["247"]["burst"].is_null()
            );
            let tags = arr(&set["cancels"]).iter().filter(|c| c["input"] == "tag").count();
            let enter = n(&set["aliases"][TAG[2]]["move"]).context("no entrance")?;
            assert!(tags > 3 && set["moves"][enter]["anim"] != "motion_261", "{tags} {}", set["moves"][enter]);
            let names = &select["quick"]["names"]["texts"];
            assert!(names[34 * 8] == "GON" && names[35 * 8 + 7] == "DOCTOR B.", "{names}");
            let mut files = Vec::new();
            for dir in ["characters", "gamedata", "gamedata/movesets", "pictures", "audio"] {
                for e in top.list(dir)?.into_iter().filter(|e| !e.is_dir) {
                    files.push((format!("{dir}/{}", e.name), top.read(&format!("{dir}/{}", e.name))?));
                }
            }
            files.sort();
            outs.push(files);
        }
        let _ = std::fs::remove_dir_all(&tmp);
        assert!(outs[0] == outs[1] && outs[0].len() > 10);
        Ok(())
    }
}
