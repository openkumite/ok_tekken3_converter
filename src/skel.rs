//! skel: pose (49 channels) -> per-row world matrices. Mirrors EXE 0x8003AD10 / 0x8003BCA8 / 0x8003AFCC.
//!
//! Works on the game's 18 matrix slots:
//!   0 root, 1 chest, 2 head, 3-6 arm A (clavicle, upper, fore, hand), 7-10 arm B,
//!   11 pelvis, 12-14 leg A (thigh, shin, foot), 15-17 leg B.
//! MESH_SLOT maps 3DMK rows to slots (EXE draw table 0x8001A074).
//! OFFSET_ROW gives the row whose joint translation feeds each slot. Joint translations are
//! stored with Z negated relative to the meshes (verified by fitting the bind pose of welded
//! seam vertices: residual ~0).

use crate::np::{self, I3, M3, V3, add, mm, mv, sub};
use crate::skin::Vert;
use anyhow::{Context, Result};
use std::collections::{BTreeSet, HashMap};
use std::f64::consts::PI;

/// (row, slot) in the Python dict's order
pub const MESH_SLOT: [(usize, usize); 19] = [
    (0, 0),
    (1, 1),
    (19, 2),
    (20, 2),
    (11, 3),
    (12, 4),
    (13, 5),
    (14, 6),
    (15, 7),
    (16, 8),
    (17, 9),
    (18, 10),
    (3, 11),
    (5, 12),
    (6, 13),
    (7, 14),
    (8, 15),
    (9, 16),
    (10, 17),
];
/// slot -> row; also SLOT_ROW ({s: r for r, s in MESH_SLOT.items() if r != 20}), which has the same values
pub const OFFSET_ROW: [usize; 18] = [0, 1, 19, 11, 12, 13, 14, 15, 16, 17, 18, 3, 5, 6, 7, 8, 9, 10];
/// slot -> parent slot (slot 0 has none)
pub const PARENT: [usize; 18] = [0, 0, 1, 1, 3, 4, 5, 1, 7, 8, 9, 0, 11, 12, 13, 11, 15, 16];
pub const MAIN_ROWS: [usize; 19] = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];

/// A world (or bind) transform: rotation, translation.
pub type Tr = (M3, V3);
/// Per row (index = row): the posed transforms.
pub type Pose = Vec<Option<Tr>>;

pub fn s32(v: u32) -> i64 {
    v as i32 as i64
}

pub fn ang(v: f64) -> f64 {
    v * 2.0 * PI / 65536.0
}

pub fn rot_zyx(ax: f64, ay: f64, az: f64) -> M3 {
    let (sx, cx) = np::sin_cos(ang(ax));
    let (sy, cy) = np::sin_cos(ang(ay));
    let (sz, cz) = np::sin_cos(ang(az));
    [
        [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
        [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
        [-sy, sx * cy, cx * cy],
    ]
}

/// Tekken 3's pose solver (0x8003AD10) in the PS1's integers: a frame's 49 channels to the 18 slots' turns on their
/// parents (rows of 4096 = 1, as the GTE keeps them). The limbs are aimed on the solver's own skeleton (0x80096ECC),
/// the same for every body: a smaller one (Gon, 45 %) bends its limbs as a full-sized one does.
pub mod solver {
    pub type M = [[i32; 3]; 3];

    const I: M = [[4096, 0, 0], [0, 4096, 0], [0, 0, 4096]];
    /// The slots the ten angle channels turn (0x8001A330).
    const ANGLE_SLOTS: [usize; 10] = [0, 1, 11, 2, 3, 7, 6, 10, 14, 17];
    /// The solver's own skeleton (0x80096ECC): the chest, the shoulders and the pelvis on their parents, then the
    /// upper limbs' starts, the targets measured from them.
    const BASE: [[i32; 3]; 8] = [
        [0, 0, 0],
        [350, 0, -100],
        [350, 0, 100],
        [0, 0, 0],
        [-140, 0, 0],
        [-140, 0, 0],
        [-160, 0, 100],
        [-160, 0, -100],
    ];
    /// 0x80096EAC: slots 2, 3 and 7 turned a quarter about y after their angles.
    const QUARTER: M = [[0, 0, -4096], [0, 4096, 0], [4096, 0, 0]];

    /// The 4096 sine table's entry `a` (12 bits).
    fn sin12(a: i32) -> i32 {
        (libm::sin((a & 0xFFF) as f64 * std::f64::consts::TAU / 4096.0) * 4096.0).round() as i32
    }

    /// The sine and cosine of an angle's top 12 bits (0x8001E8C4, `(a >> 3) & 0x1FFE`).
    fn sc(a: i32) -> (i32, i32) {
        let k = (a & 0xFFFF) >> 4;
        (sin12(k), sin12(k + 1024))
    }

    /// The GTE's square root.
    fn gte_sqrt(a: i32) -> i32 {
        if a <= 0 {
            return 0;
        }
        let lz = (a as u32).leading_zeros() & !1;
        let m = if lz <= 24 { (a as u32) >> (24 - lz) } else { (a as u32) << (lz - 24) } as i32;
        let t = (f64::from(m) / 64.0).sqrt() * 4096.0;
        ((t as i32) << ((31 - lz) >> 1)) >> 12
    }

    fn s16(v: i32) -> i32 {
        v as i16 as i32
    }

    /// MulMatrix (0x80082288): a x b, each sum >> 12, kept to the GTE's s16.
    fn mul(a: &M, b: &M) -> M {
        std::array::from_fn(|r| {
            std::array::from_fn(|c| ((0..3).map(|k| a[r][k] * b[k][c]).sum::<i32>() >> 12).clamp(-0x8000, 0x7FFF))
        })
    }

    /// MVMVA (sf 1, the MAC): a v >> 12.
    fn mv(a: &M, v: [i32; 3]) -> [i32; 3] {
        std::array::from_fn(|r| ((0..3).map(|k| i64::from(a[r][k]) * i64::from(v[k])).sum::<i64>() >> 12) as i32)
    }

    /// 0x8003A338: Rz Ry Rx of the s16 angles, each product >> 12 in its order.
    fn euler(x: i32, y: i32, z: i32) -> M {
        let ((sx, cx), (sy, cy), (sz, cz)) = (sc(x), sc(y), sc(z));
        let cxcz = (cx * cz) >> 12;
        let (sxcz, sxsz, sxcy, cxsz) = ((sx * cz) >> 12, (sx * sz) >> 12, (sx * cy) >> 12, (cx * sz) >> 12);
        let (a, b, c) = ((sy * sxcz) >> 12, (sy * sxsz) >> 12, (sy * cxcz) >> 12);
        let m = [
            [(cy * cz) >> 12, a - cxsz, sxsz + c],
            [(cy * sz) >> 12, cxcz + b, ((cxsz * sy) >> 12) - sxcz],
            [-sy, sxcy, (cy * cx) >> 12],
        ];
        m.map(|r| r.map(s16))
    }

    /// 0x8003BA28: `target` in the axes of `(turn, at)` (its transpose times the difference, split at bit 15 for
    /// the GTE: the high part unshifted times 8, the low part >> 12), plus `off`.
    fn toward(target: [i32; 3], turn: &M, at: [i32; 3], off: [i32; 3]) -> [i32; 3] {
        let d: [i32; 3] = std::array::from_fn(|k| target[k] - at[k]);
        let part = |v: i32, hi: bool| {
            let a = if hi { v.abs() >> 15 } else { v.abs() & 0x7FFF };
            if v < 0 { -a } else { a }
        };
        std::array::from_fn(|r| {
            let hi: i32 = (0..3).map(|k| turn[k][r] * part(d[k], true)).sum();
            let lo: i32 = (0..3).map(|k| turn[k][r] * part(d[k], false)).sum::<i32>() >> 12;
            let hi = if hi < 0 { -((-hi) << 3) } else { hi << 3 };
            lo + hi + off[r]
        })
    }

    /// A sum >> 12 cut toward zero (0x8003C148's `addiu 0xFFF`).
    fn cut(v: i32) -> i32 {
        (v + if v < 0 { 0xFFF } else { 0 }) >> 12
    }

    /// 0x8003BCA8: an arm's (`elbow`) or a leg's two slots aimed at the target `v` (the upper's axes), the upper
    /// turned by `twist` about its length, the joint bent by the law of cosines on the solver's own lengths (as the
    /// constants fold them in). `lower` is kept when the target sits on the joint.
    fn limb(v: [i32; 3], twist: i32, elbow: bool, lower: M) -> Option<(M, M)> {
        let [x, y, z] = v;
        let d = x.wrapping_mul(x).wrapping_add(y.wrapping_mul(y)).wrapping_add(z.wrapping_mul(z));
        if d == 0 {
            return None;
        }
        let l = gte_sqrt(d);
        let inv = 0x40_0000 / l.max(1);
        let (nx, ny, nz) = ((x * inv) >> 10, (y * inv) >> 10, (z * inv) >> 10);
        let (t1, t2, a3) = match nx + 0x1000 {
            0 => (-0x800, (y * inv) >> 11, -0x1000),
            k if k >= 0x2000 => (-0x800, (-ny) >> 1, 0x1000),
            k => {
                let t1 = -0x100_0000 / k;
                (t1, (t1 * ny) >> 12, 0x1000)
            }
        };
        let a1 = ((t2 * ny) >> 12) + a3;
        let v0 = (t1 * nz) >> 12;
        let a3 = a3 + nx;
        let (t0, a2, t5, t8, t9) =
            ((t2 * a3) >> 12, (v0 * a3) >> 12, (v0 * ny) >> 12, (t2 * nz) >> 12, ((v0 * nz) >> 12) + 0x1000);
        let (s, c) = sc(twist);
        let upper = [
            [nx, (c * t0 + s * a2) >> 12, (-s * t0 + c * a2) >> 12],
            [ny, (c * a1 + s * t5) >> 12, (-s * a1 + c * t5) >> 12],
            [nz, (c * t8 + s * t9) >> 12, (-s * t8 + c * t9) >> 12],
        ]
        .map(|r| r.map(s16));
        let (far, near, k, ka, kb) = if elbow { (0x216, 0x47, 37380, 217, 282) } else { (0x37C, 3, 1784, 146, 147) };
        if l >= far {
            return Some((upper, I));
        }
        if l < near {
            return Some((upper, lower));
        }
        let (a, b) = (l << 11, (inv * k) >> 11);
        let unit = -0x1000..=0x1000;
        let cos = ((a + b) * ka) >> 16;
        let sin =
            if unit.contains(&cos) { s16(gte_sqrt(0x100_0000 - cos * cos) * if elbow { 1 } else { -1 }) } else { 0 };
        let cos = cos.clamp(-0x1000, 0x1000);
        // 0x8003BB88: the upper's first two columns turned by the bend
        let upper = upper.map(|r| [s16((r[0] * cos + r[1] * sin) >> 12), s16((-r[0] * sin + r[1] * cos) >> 12), r[2]]);
        let c2 = ((a - b) * kb) >> 16;
        let s2 = if unit.contains(&c2) { (-sin * if elbow { 5332 } else { 4114 }) >> 12 } else { 0 };
        let c2 = c2.clamp(-0x1000, 0x1000);
        let (cc, ss) = (cut(cos * c2 + sin * s2), cut(-sin * c2 + cos * s2));
        Some((upper, [[cc, -ss, 0], [ss, cc, 0], [0, 0, 4096]].map(|r| r.map(s16))))
    }

    /// The 18 slots' turns on their parents for a frame's 49 channels (each an s16).
    pub fn locals(p: &[i64]) -> [M; 18] {
        let ch = |k: usize| p.get(k).copied().unwrap_or(0) as i16 as i32;
        let mut l = [I; 18];
        for (i, &s) in ANGLE_SLOTS.iter().enumerate() {
            l[s] = euler(ch(19 + 3 * i), ch(20 + 3 * i), ch(21 + 3 * i));
        }
        for s in [2, 3, 7] {
            l[s] = mul(&l[s], &QUARTER);
        }
        // the solver's frames: the root at the motion's place, the chest, the shoulders and the pelvis on its skeleton
        let on = |p: &(M, [i32; 3]), r: &M, t: [i32; 3]| -> (M, [i32; 3]) {
            let at = mv(&p.0, t);
            (mul(&p.0, r), std::array::from_fn(|k| at[k] + p.1[k]))
        };
        let root = (l[0], [ch(0), ch(1), ch(2)]);
        let chest = on(&root, &l[1], BASE[0]);
        let shoulders = [on(&chest, &l[3], BASE[1]), on(&chest, &l[7], BASE[2])];
        let pelvis = on(&root, &l[11], BASE[3]);
        let limbs = [(&shoulders[0], 4, true), (&shoulders[1], 8, true), (&pelvis, 12, false), (&pelvis, 15, false)];
        for (k, (base, upper, elbow)) in limbs.into_iter().enumerate() {
            let v = toward([ch(3 + 3 * k), ch(4 + 3 * k), ch(5 + 3 * k)], &base.0, base.1, BASE[4 + k]);
            if let Some((u, low)) = limb(v, ch(15 + k), elbow, l[upper + 1]) {
                (l[upper], l[upper + 1]) = (u, low);
            }
        }
        l
    }
}

/// An accessory bone's neutral placement (see fit_extras): fitted bind transform, gravity-hung, or rigid.
#[derive(Clone)]
pub enum Mode {
    Fit(M3, V3),
    Hang(V3),
    Rigid(V3),
}

#[derive(Clone)]
pub struct Extra {
    pub mode: Mode,
    pub r#ref: usize,
}

impl Extra {
    pub fn t(&self) -> V3 {
        match self.mode {
            Mode::Fit(_, t) | Mode::Hang(t) | Mode::Rigid(t) => t,
        }
    }
}

/// row -> extra, in the Python dict's insertion order
pub type Extras = Vec<(usize, Extra)>;

pub fn get_extra(e: &Extras, r: usize) -> Option<&Extra> {
    e.iter().find(|x| x.0 == r).map(|x| &x.1)
}

pub struct Skeleton {
    pub rows: Vec<[u32; 14]>,
    pub t: Vec<V3>,
    pub parent: Vec<Option<usize>>,
    pub extras: Extras,
}

impl Skeleton {
    pub fn new(model: &[u8], extras: Extras) -> Result<Skeleton> {
        let (rows, _) = crate::t3m::parse(model)?;
        let n = rows.len();
        let t = rows.iter().map(|r| [s32(r[3]) as f64, s32(r[4]) as f64, -s32(r[5]) as f64]).collect();
        let bones: Vec<usize> = (0..n).filter(|i| ![2, 4, 20].contains(i)).collect();
        let parent = rows
            .iter()
            .map(|r| match usize::try_from(s32(r[6])) {
                Ok(p) => bones.get(p).copied().map(Some).context("a bone's parent is out of range"),
                Err(_) => Ok(None),
            })
            .collect::<Result<_>>()?;
        Ok(Skeleton { rows, t, parent, extras })
    }

    fn row_t(&self, r: usize) -> Result<V3> {
        self.t.get(r).copied().with_context(|| format!("the model has no row {r}"))
    }

    /// P: 49 channels. Returns per row world transforms (character space, y down).
    /// springs: a Springs state to advance one frame (accessory links); else they hold neutral.
    pub fn pose(&self, p: &[i64], springs: Option<&mut Springs>) -> Result<Pose> {
        if p.len() < 49 {
            anyhow::bail!("a pose has fewer than 49 channels");
        }
        let t = |s: usize| self.row_t(OFFSET_ROW[s]);
        // the slots' turns from the PS1's pose solver, its own skeleton's limbs for every body (Gon's legs too)
        let l = solver::locals(p).map(|m| m.map(|r| r.map(|v| v as f64 / 4096.0)));
        let mut w: [Option<Tr>; 18] = Default::default();
        w[0] = Some((l[0], [p[0] as f64, p[1] as f64, p[2] as f64]));
        for s in 1..18 {
            let (pr, pt) = w[PARENT[s]].context("a slot's parent is not posed")?;
            w[s] = Some((mm(&pr, false, &l[s], false), add(&mv(&pr, false, &t(s)?), &pt)));
        }
        let n = self.rows.len();
        let mut out: Pose = vec![None; n.max(21)];
        for (r, s) in MESH_SLOT {
            if r < n {
                out[r] = w[s];
            }
        }
        if let Some(sp) = springs {
            sp.step(self, &mut out)?;
        }
        // accessory bones (see fit_extras): fitted bind transform, gravity-hung, or rigid
        for _ in 0..8 {
            for (r, e) in &self.extras {
                if out.get(*r).is_none_or(|o| o.is_some()) {
                    continue;
                }
                let Some((rr, tt)) = out.get(e.r#ref).copied().flatten() else { continue };
                out[*r] = Some(match &e.mode {
                    Mode::Fit(er, et) => (mm(&rr, false, er, false), add(&mv(&rr, false, et), &tt)),
                    Mode::Rigid(et) => (rr, add(&mv(&rr, false, et), &tt)),
                    // hang: rotate the bone so its +X axis points down (+Y is down in PS1 space)
                    Mode::Hang(et) => (hang(&rr), add(&mv(&rr, false, et), &tt)),
                });
            }
        }
        for _ in 0..6 {
            // remaining chain links follow their parent rigidly
            for r in 0..n {
                if out[r].is_some() {
                    continue;
                }
                if let Some((rr, tt)) = self.parent[r].and_then(|p| out[p]) {
                    out[r] = Some((rr, add(&mv(&rr, false, &self.t[r]), &tt)));
                }
            }
        }
        Ok(out)
    }
}

/// Minimal rotation taking the bone's +X axis to straight down, applied on top of R.
fn hang(r: &M3) -> M3 {
    let x = [r[0][0], r[1][0], r[2][0]];
    let dn = [0.0, 1.0, 0.0];
    let c = np::dot(&x, &dn);
    let ax = [x[1] * dn[2] - x[2] * dn[1], x[2] * dn[0] - x[0] * dn[2], x[0] * dn[1] - x[1] * dn[0]];
    let s = np::norm(&ax);
    if s < 1e-6 {
        return if c > 0.0 { *r } else { mm(&[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], false, r, false) };
    }
    let u = [ax[0] / s, ax[1] / s, ax[2] / s];
    let k = [[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]];
    let kk = mm(&k, false, &k, false);
    let q: M3 = std::array::from_fn(|i| std::array::from_fn(|j| (I3[i][j] + s * k[i][j]) + (1.0 - c) * kk[i][j]));
    mm(&q, false, r, false)
}

/// A row of fit_extras's input: [parent, tx, ty, tz, has_geo, ax, ay, az].
pub type ExtraRow = [i64; 8];

/// Neutral transforms for accessory bones (hair, tails, dress panels, coats...).
/// rows: [parent, tx, ty, tz, has_geo, ax, ay, az]; verts: skin::build() output.
/// Rows 2 and 4 (extra sections such as jackets) ride on the chest and pelvis. Accessory slots
/// (rows 21+) carry their neutral Euler angles in row words 7-9; the game loads them into the bone's
/// local matrix (EXE 0x80037C5C) and runs a spring simulation around them (0x80037CBC, see
/// Springs). Remaining extras are placed by fitting welded seam vertices (Kabsch), else hang/rigid.
pub fn fit_extras(rows: &[ExtraRow], verts: &[Vert]) -> Result<Extras> {
    let pairs = weld_pairs(verts);
    let n = rows.len();
    let bones: Vec<usize> = (0..n).filter(|i| ![2, 4, 20].contains(i)).collect();
    let bone = |p: i64| bones.get(p as usize).copied().context("an accessory's parent is out of range");
    let extra: Vec<usize> = (0..n).filter(|&i| rows[i][4] != 0 && !MAIN_ROWS.contains(&i)).collect();
    // a set of small ints: Python iterates it in ascending order
    let mut placed: BTreeSet<usize> = MAIN_ROWS.into_iter().collect();
    let mut out: Extras = Vec::new();
    let has = |out: &Extras, r: usize| out.iter().any(|e| e.0 == r);
    // Rows 2/4/20 are drawn as secondary meshes of the section before them (chest, pelvis, head),
    // in that bone's local frame.
    for (r, rf) in [(2, 1), (4, 3)] {
        if extra.contains(&r) {
            out.push((r, Extra { mode: Mode::Fit(I3, [0.0; 3]), r#ref: rf }));
            placed.insert(r);
        }
    }
    for &r in &extra {
        if r >= 21 && rows[r][0] >= 0 {
            let s16 = |v: i64| if v >= 32768 { v - 65536 } else { v } as f64;
            let rr = rot_zyx(s16(rows[r][5]), s16(rows[r][6]), s16(rows[r][7]));
            let t = [rows[r][1] as f64, rows[r][2] as f64, -rows[r][3] as f64];
            out.push((r, Extra { mode: Mode::Fit(rr, t), r#ref: bone(rows[r][0])? }));
        }
    }
    for _ in 0..8 {
        for &r in &extra {
            if has(&out, r) {
                continue;
            }
            let mut best: Option<(usize, Fit)> = None;
            for &q in &placed {
                if let Some(f) = pairs.get(&(q, r)).and_then(|p| kabsch(p))
                    && f.2 < 8.0
                    && best.as_ref().is_none_or(|b| f.3 > b.1.3)
                {
                    best = Some((q, f));
                }
            }
            if let Some((q, f)) = best {
                out.push((r, Extra { mode: Mode::Fit(f.0, f.1), r#ref: q }));
                placed.insert(r);
            }
        }
    }
    for &r in &extra {
        if has(&out, r) {
            continue;
        }
        let p = rows[r][0];
        let par = if p >= 0 { bone(p)? } else { 0 };
        let t = [rows[r][1] as f64, rows[r][2] as f64, -rows[r][3] as f64];
        out.push((r, Extra { mode: if par != 14 && par != 18 { Mode::Hang(t) } else { Mode::Rigid(t) }, r#ref: par }));
    }
    Ok(out)
}

/// A local position pair: (point in bone A's frame, the same point in bone B's frame).
pub type Pair = (V3, V3);

/// {(rowA,rowB): [(pA_local, pB_local), ...]} for vertices blended between two bones.
pub fn weld_pairs(verts: &[Vert]) -> HashMap<(usize, usize), Vec<Pair>> {
    let mut pairs: HashMap<(usize, usize), Vec<Pair>> = HashMap::new();
    for v in verts.iter().flatten() {
        let mut d: Vec<(usize, V3)> = Vec::new();
        for (r, _, p) in v.iter() {
            if !d.iter().any(|e| e.0 == *r) {
                d.push((*r, p.map(|c| c as f64)));
            }
        }
        for a in &d {
            for b in &d {
                if a.0 != b.0 {
                    pairs.entry((a.0, b.0)).or_default().push((a.1, b.1));
                }
            }
        }
    }
    pairs
}

/// kabsch's result: R, t, max residual, n
pub type Fit = (M3, V3, f64, usize);

/// Fit pA = R pB + t. None if fewer than 3 pairs, or all on one line.
pub fn kabsch(p: &[Pair]) -> Option<Fit> {
    let n = p.len();
    if n < 3 {
        return None;
    }
    let mean = |k: usize| -> V3 {
        let mut s = [0.0; 3];
        for (i, e) in p.iter().enumerate() {
            let v = if k == 0 { e.0 } else { e.1 };
            s = if i == 0 { v } else { add(&s, &v) };
        }
        s.map(|c| c / n as f64)
    };
    let (ca, cb) = (mean(0), mean(1));
    let a0: Vec<f64> = p.iter().flat_map(|e| sub(&e.0, &ca)).collect();
    let b0: Vec<f64> = p.iter().flat_map(|e| sub(&e.1, &cb)).collect();
    let h = np::gemm(&b0, true, &a0, false, 3, 3, n);
    let (u, s, vt) = np::svd(&[[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]]);
    // pairs on one line (often two points repeated) leave the turn about it free: the SVD's pick, not a fit
    if s[1] <= s[0] * 1e-9 {
        return None;
    }
    let m = mm(&vt, true, &u, true);
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let sign = if det > 0.0 {
        1.0
    } else if det < 0.0 {
        -1.0
    } else {
        0.0
    };
    let d = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, sign]];
    let r = mm(&mm(&vt, true, &d, false), false, &u, true);
    let t = sub(&ca, &mv(&r, false, &cb));
    let bs: Vec<f64> = p.iter().flat_map(|e| e.1).collect();
    let rb = np::gemm(&bs, false, &[r[0], r[1], r[2]].concat(), true, n, 3, 3);
    let res = p
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let q = sub(&e.0, &add(&[rb[3 * i], rb[3 * i + 1], rb[3 * i + 2]], &t));
            (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt()
        })
        .fold(f64::NEG_INFINITY, f64::max);
    Some((r, t, res, n))
}

// ---- accessory springs (hair, tails, coat tails, skirts) -------------------------------------
// Port of EXE 0x80037CBC, run by the pose solver (0x8003B418) once per frame for slots 18-21,
// which hold rows 21-24. Each link keeps its tip from the previous frame; the new direction is a
// blend of that lagging direction, gravity and the neutral (rest) direction, turned into two
// angles on top of the neutral rotation. Parameters are per character (index = player id, which
// is the model's position in the BNS archive: (record - 71) / 4).
pub const EXE_BASE: i64 = 0x80010000 - 0x800; // PS-X EXE: file offset = address - EXE_BASE
pub const SPRING_TABLE: i64 = 0x8009623C;
pub const SPRING_NONE: u32 = 0x80095F48;
pub const SPRING_SIGNS: i64 = 0x80095D20;

/// 0x80095D10 / 0x80095D14: (hi, lo)
fn spring_clamp(c: i64) -> Option<(f64, f64)> {
    match c {
        26 => Some((4096.0, -4096.0)),
        36 | 41 => Some((8192.0, -8192.0)),
        _ => None,
    }
}

/// 4096 units per turn
pub fn ratan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x) * 4096.0 / (2.0 * PI)
}

/// Python's `math.hypot(x, y)` (3.12: a correctly rounded vector norm, not the C library's hypot).
pub fn hypot(x: f64, y: f64) -> f64 {
    let (x, y) = (x.abs(), y.abs());
    let max = x.max(y);
    if max.is_infinite() || max == 0.0 {
        return max;
    }
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    let max_e = frexp_exp(max);
    if max_e < -1023 {
        return f64::MIN_POSITIVE * hypot(x / f64::MIN_POSITIVE, y / f64::MIN_POSITIVE);
    }
    let scale = 2f64.powi(-max_e);
    let (mut csum, mut frac1, mut frac2) = (1.0f64, 0.0f64, 0.0f64);
    let dl_mul = |a: f64, b: f64| {
        let z = a * b;
        (z, ok_converter_common::np::fma(a, b, -z))
    };
    let dl_fast_sum = |a: f64, b: f64| {
        let x = a + b;
        (x, b - (x - a))
    };
    for v in [x, y] {
        let v = v * scale;
        let pr = dl_mul(v, v);
        let sm = dl_fast_sum(csum, pr.0);
        csum = sm.0;
        frac1 += pr.1;
        frac2 += sm.1;
    }
    let mut h = (csum - 1.0 + (frac1 + frac2)).sqrt();
    let pr = dl_mul(-h, h);
    let sm = dl_fast_sum(csum, pr.0);
    csum = sm.0;
    frac1 += pr.1;
    frac2 += sm.1;
    let x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    h / scale
}

/// frexp's exponent: x = m * 2^e with 0.5 <= m < 1
fn frexp_exp(x: f64) -> i32 {
    let b = x.to_bits();
    let e = ((b >> 52) & 0x7ff) as i32;
    if e == 0 {
        // subnormal
        frexp_exp(x * 2f64.powi(64)) - 64
    } else {
        e - 1022
    }
}

#[derive(Clone)]
pub struct Spring {
    pub typ: i64,
    pub w_lag: i64,
    pub w_grav: i64,
    pub len: i64,
    pub grav: i64,
    pub collide: bool,
    pub coll_slot: i64,
    pub coll_point: V3,
    pub coll_cmp: i64,
    pub signs: Vec<[i64; 6]>,
}

/// Spring parameters for accessory links 0-3 of a character (None = no spring).
pub fn spring_params(exe: &[u8], char: i64) -> Result<Vec<Option<Spring>>> {
    let rd = |a: i64, n: usize| -> Result<&[u8]> {
        usize::try_from(a - EXE_BASE)
            .ok()
            .and_then(|o| exe.get(o..o + n))
            .context("a spring table read is outside the EXE")
    };
    let i32s = |b: &[u8]| b.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c) as i64).collect::<Vec<_>>();
    let signs = (0..10)
        .map(|i| -> Result<[i64; 6]> {
            let v = i32s(rd(SPRING_SIGNS + 24 * i, 24)?);
            Ok([v[0], v[1], v[2], v[3], v[4], v[5]])
        })
        .collect::<Result<Vec<_>>>()?;
    let mut out = Vec::new();
    for c in rd(SPRING_TABLE + 24 * char, 16)?.as_chunks::<4>().0 {
        let p = u32::from_le_bytes(*c);
        if p == SPRING_NONE {
            out.push(None);
            continue;
        }
        let v: Vec<i64> = rd(p as i64, 24)?.as_chunks::<2>().0.iter().map(|c| i16::from_le_bytes(*c) as i64).collect();
        out.push(Some(Spring {
            typ: v[0],
            w_lag: v[2],
            w_grav: v[3],
            len: v[4],
            grav: v[5],
            collide: v[6] == 1,
            coll_slot: v[7],
            coll_point: [v[8] as f64, v[9] as f64, v[10] as f64],
            coll_cmp: v[11],
            signs: signs.clone(),
        }));
    }
    Ok(out)
}

/// Python's `l[i]` for a possibly negative i
fn py_get<T>(l: &[T], i: i64) -> Result<&T> {
    let k = if i < 0 { i + l.len() as i64 } else { i };
    usize::try_from(k).ok().and_then(|k| l.get(k)).context("a spring parameter indexes past its table")
}

/// Per-clip simulation state. step() runs after the main skeleton is posed.
pub struct Springs {
    pub char: i64,
    /// (link, row, params, neutral rotation)
    pub links: Vec<(usize, usize, Option<Spring>, M3)>,
    pub prev: Vec<Option<Tr>>,
}

impl Springs {
    pub fn new(sk: &Skeleton, params: &[Option<Spring>], char: i64) -> Springs {
        let mut links = Vec::new();
        for (s, p) in params.iter().enumerate() {
            let r = 21 + s;
            let Some(row) = sk.rows.get(r) else { continue };
            if row[10] == 0 {
                continue;
            }
            let a = |v: u32| {
                let v = (v & 0xffff) as i64;
                (if v >= 32768 { v - 65536 } else { v }) as f64
            };
            links.push((s, r, p.clone(), rot_zyx(a(row[7]), a(row[8]), a(row[9]))));
        }
        Springs { char, links, prev: vec![None; sk.rows.len()] }
    }

    fn angles(v: &V3, e: &[i64; 6]) -> (f64, f64) {
        let yaw = ratan2(e[0] as f64 * v[2], e[1] as f64 * v[0]) * e[4] as f64;
        let pitch = ratan2(e[2] as f64 * v[1], e[3] as f64 * hypot(v[0], v[2])) * e[5] as f64;
        (yaw, pitch)
    }

    pub fn step(&mut self, sk: &Skeleton, out: &mut Pose) -> Result<()> {
        let c = self.char;
        let get = |out: &Pose, r: usize| out.get(r).copied().flatten();
        for (s, r, p, rn) in &self.links {
            let (s, r) = (*s, *r);
            let Some((pr, pt)) = sk.parent[r].and_then(|q| get(out, q)) else { continue };
            let t = sk.t[r];
            let live = p.is_some() && !((c == 32 || c == 33) && s < 2) && self.prev[r].is_some();
            let (Some(p), true) = (p, live) else {
                // first frame / no spring: neutral pose
                out[r] = Some((mm(&pr, false, rn, false), add(&mv(&pr, false, &t), &pt)));
                self.prev[r] = out[r];
                continue;
            };
            let b = mm(&pr, false, rn, false); // neutral frame; the link points along +X
            let (p_r, p_t) = self.prev[r].context("a spring without its previous frame")?;
            let mut tip = add(&mv(&p_r, false, &[p.len as f64, 0.0, 0.0]), &p_t);
            let pivot = add(&mv(&pr, false, &t), &pt);
            let airborne = tip[1] < 0.0; // y is down; the floor is y = 0
            if !airborne {
                tip[1] = 0.0;
            }
            let d = mv(&b, true, &sub(&tip, &pivot));
            let n = np::norm(&d);
            let a = if n > 1e-9 { d.map(|x| x / n * 4096.0) } else { [4096.0, 0.0, 0.0] };
            let k = p.grav;
            let g = if airborne {
                mv(&b, true, &[(k >> 4) as f64, ((256 - k) << 4) as f64, (k >> 4) as f64])
            } else {
                [0.0; 3]
            };
            let mut v: V3 = std::array::from_fn(|i| (a[i] * p.w_lag as f64 + g[i] * p.w_grav as f64) / 256.0);
            v[0] += ((256 - p.w_lag - p.w_grav) * 16) as f64;
            let (mut typ, mut flag, mut lim) = (p.typ, false, None);
            if p.collide {
                // keep the link off a thigh (0x80038430)
                let slot = usize::try_from(p.coll_slot).ok().and_then(|s| OFFSET_ROW.get(s));
                let (lr, lt) = slot.and_then(|&q| get(out, q)).context("a spring collides with an unposed bone")?;
                let q = mv(&b, true, &sub(&add(&mv(&lr, false, &p.coll_point), &lt), &pivot));
                let mut lt = typ;
                if c == 38 && q[0] < 0.0 {
                    (lt, flag) = (8, true);
                }
                lim = Some(Self::angles(&q, py_get(&p.signs, lt)?));
            } else if (c == 36 || c == 41) && s == 0 {
                // front dress panel aims between the knees (0x8003866c)
                let ka = get(out, OFFSET_ROW[13]).context("knee A")?.1;
                let kb = get(out, OFFSET_ROW[16]).context("knee B")?.1;
                let dk = sub(&kb, &ka);
                let l = np::norm(&dk);
                if l * l >= 16384.0 {
                    let mut pt = add(&dk.map(|x| x * 128.0 / l), &ka);
                    pt[2] += 64.0;
                    let w = mv(&b, true, &sub(&pt, &p_t));
                    let nw = np::norm(&w);
                    v = w.map(|x| x / nw * 4096.0);
                }
            }
            if c == 38 && flag {
                typ = 2;
            }
            let (mut yaw, mut pitch) = Self::angles(&v, py_get(&p.signs, typ)?);
            if let Some(lim) = lim {
                let e = py_get(&p.signs, p.coll_cmp)?;
                let (a, b) = (e[5] as f64, e[4] as f64);
                if a * lim.0 + b * lim.1 < a * yaw + b * pitch {
                    (yaw, pitch) = lim;
                }
            }
            (yaw, pitch) = (yaw * 16.0, pitch * 16.0); // -> 65536 units per turn
            if [(14, 0), (27, 2), (38, 3)].contains(&(c, s)) || (c == 15 && (s == 1 || s == 2)) {
                yaw = 0.0;
            }
            if c == 38 && s == 0 && pitch < 2048.0 {
                pitch = 2048.0;
            }
            if let Some((hi, lo)) = spring_clamp(c)
                && (c == 26 || s < 4)
            {
                yaw = yaw.max(lo).min(hi);
                pitch = pitch.max(lo).min(hi);
            }
            out[r] = Some((mm(&b, false, &rot_zyx(0.0, yaw, pitch), false), pivot));
            self.prev[r] = out[r];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// math.hypot of Python 3.12 (values from `python3 -c 'import math; print(math.hypot(x, y).hex())'`)
    #[test]
    fn hypot_as_python() {
        for (x, y, want) in
            [(1e-310, 3e-310, 0x00003a365ff2ea11), (3.7, -1.1, 0x400ee162d851db37), (0.1, 0.2, 0x3fcc9f25c5bfedda)]
        {
            assert_eq!(hypot(x, y).to_bits(), want, "{x} {y}");
        }
    }

    /// two seam points, one repeated: the rotation about their line is anyone's guess, so no fit
    #[test]
    fn kabsch_on_a_line_is_no_fit() {
        let (a, b) = ([73.0, -6.0, -74.0], [39.0, 54.0, 97.0]);
        assert!(kabsch(&[(a, a), (b, b), (a, a)]).is_none());
        assert!(kabsch(&[(a, a), (b, b), ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])]).is_some());
    }

    /// King's frame as the PS1 logged it (PCSX-Redux, VS, P+0xF74 at the facing driver): the solver's turns.
    #[test]
    fn kings_pose_as_on_the_ps1() {
        let c: [i64; 49] = [
            44, -904, -126, -230, -1164, -448, 178, -1166, -710, -80, -186, 184, 176, -112, -446, 27596, 3720, 1194,
            306, 0, 0, 0, -22850, 2892, -17696, 23850, 1052, 16288, 11648, 342, 30426, -31712, -1262, 16356, 400, -76,
            16304, -1908, -442, 1342, -3690, -834, 36, 598, -922, -22384, -2244, -1108, -20482,
        ];
        let l = solver::locals(&c);
        assert_eq!(l[4], [[22, 2554, -3224], [-2067, -2758, -2208], [-3570, 1615, 1336]]);
        assert_eq!(l[5], [[-508, 4051, 0], [-4051, -508, 0], [0, 0, 4096]]);
        assert_eq!(l[13], [[-281, -4078, 0], [4078, -281, 0], [0, 0, 4096]]);
        assert_eq!(l[17], [[-1564, 3658, 975], [-3760, -1624, 59], [439, -874, 3977]]);
    }

    /// T3_BIN=<the disc image>: Gon's legs (45 %) where the PS1 puts them (PCSX-Redux, VS, Gon against Paul, after the
    /// skeleton 0x8003AFCC: the 49 channels at 0x800A85E8, slot 0's matrix and place and slots 12-17's places at
    /// P+0x8F4): his stance, walking forward and back, the crouch, his roll, f+4, d+4 and 3. Each slot's place from
    /// slot 0, in its axes, within the PS1's own rounding (his own limbs' lengths put them 126 units off).
    #[test]
    fn gons_legs_as_on_the_ps1() -> Result<()> {
        let Ok(bin) = std::env::var("T3_BIN") else {
            return Ok(());
        };
        let game = crate::disc::Game::load(&crate::disc::track1(bin)?)?;
        let sk = Skeleton::new(game.rec.get(239).context("no Gon")?, Vec::new())?;
        type Row = ([i64; 49], [i64; 12], [[i64; 3]; 6]);
        let rows: [Row; 8] = [
            // vsync 1720, frame 9
            (
                [
                    0, -988, 0, -222, -878, -336, 180, -880, -396, -204, -136, -26, 202, -138, -92, 15644, 15114, 4826,
                    -8218, 0, 0, 0, -18096, 5908, -16382, 18420, 0, 16382, 14862, -826, 27566, 32766, 0, 16384, 0, 0,
                    16384, -7250, 90, 3632, 9054, -78, 5462, 832, 532, -21580, -726, 3828, -21602,
                ],
                [0, 0, -4096, 0, 4096, 0, 4096, 0, 0, -793, -445, -1],
                [
                    [-802, -374, -45],
                    [-695, -240, -143],
                    [-783, -71, -93],
                    [-785, -375, 43],
                    [-669, -244, 132],
                    [-754, -72, 87],
                ],
            ),
            // vsync 1905, frame 2
            (
                [
                    0, -990, -30, -224, -876, -374, 178, -878, -426, -206, -150, -30, 200, -144, -140, 15858, 14890,
                    4278, -8758, 0, 0, 0, -18078, 6286, -16382, 19120, 0, 16384, 14932, -854, 27134, -32768, 0, 16384,
                    0, 0, 16384, -7250, 90, 3632, 9056, -78, 5462, 1140, 354, -21378, -726, 3706, -21586,
                ],
                [-6, 0, -4096, 0, 4096, 0, 4096, 0, -6, 658, -446, 3],
                [
                    [646, -375, -40],
                    [750, -241, -142],
                    [657, -76, -89],
                    [669, -375, 46],
                    [789, -247, 134],
                    [705, -73, 90],
                ],
            ),
            // vsync 2001, frame 3
            (
                [
                    0, -1094, 150, -246, -1044, -192, 152, -1050, -268, -48, -146, -150, 44, -570, 958, 15696, 15018,
                    3470, 3586, 0, 0, 0, -18780, 4418, -16382, 12558, -2864, 16384, 14498, -1460, 25524, -32768, 0,
                    16384, 0, 0, 16384, -7250, 90, 3632, 9056, -78, 5462, -2020, 3678, -19648, 560, -16, -18688,
                ],
                [0, 0, -4096, 0, 4096, 0, 4096, 0, 0, -250, -493, 2],
                [
                    [-255, -421, -40],
                    [-126, -272, -29],
                    [-120, -76, -21],
                    [-285, -430, 44],
                    [-410, -277, 32],
                    [-608, -264, 21],
                ],
            ),
            // vsync 2100, frame 30
            (
                [
                    0, -818, 0, -170, -808, -568, 120, -896, -738, -196, -120, 306, 204, -112, -412, 29284, -80, 1386,
                    -6748, 0, 0, 0, -17782, 12450, -16382, 21846, 0, 16384, 15812, -1338, 20862, 32528, -1268, 10082,
                    3364, 1606, 9126, 0, 0, -142, 0, 0, -142, 4332, 0, -27618, -1068, 722, -17790,
                ],
                [0, 0, -4096, 0, 4096, 0, 4096, 0, 0, -1647, -369, 2],
                [
                    [-1669, -298, -37],
                    [-1618, -146, -154],
                    [-1783, -61, -87],
                    [-1625, -298, 40],
                    [-1437, -252, 74],
                    [-1467, -58, 90],
                ],
            ),
            // vsync 2230, frame 28
            (
                [
                    1322, -694, -1048, 1006, -324, -1434, 1824, -592, -1502, 1368, -204, -792, 1554, -110, -1506,
                    21726, 10126, 1684, -4630, 1738, 0, -92, -17984, 10588, -19122, 17728, 926, 16476, 16988, 12,
                    32182, -32614, -5086, 11442, 2974, -152, 10250, 74, 4746, -164, 38, -18, -118, 358, 2708, -24644,
                    1488, 2828, -14724,
                ],
                [1573, -609, -3733, -38, 4040, -675, 3782, 294, 1545, -1090, -313, 471],
                [
                    [-1117, -243, 435],
                    [-1041, -63, 404],
                    [-1187, -98, 532],
                    [-1073, -242, 512],
                    [-878, -253, 493],
                    [-866, -56, 487],
                ],
            ),
            // vsync 2345, frame 23
            (
                [
                    94, -1108, -856, -348, -938, -830, -36, -1422, -1062, -108, -1340, -1776, -18, -114, -834, 14280,
                    8526, 15214, 8172, 0, -13520, 0, -29758, -7902, -13340, 15180, 2144, 28040, 20960, -4228, -25572,
                    32766, 0, 16384, 0, 0, 16384, 0, 96, -8, 0, -90, 0, 1542, 1122, -16110, -376, -5254, -20322,
                ],
                [-3696, 0, -1769, 0, 4096, 0, 1768, 0, -3696, -688, -499, 459],
                [
                    [-607, -509, 441],
                    [-472, -646, 394],
                    [-304, -603, 300],
                    [-645, -428, 448],
                    [-622, -233, 459],
                    [-708, -61, 411],
                ],
            ),
            // vsync 2470, frame 26
            (
                [
                    -4, -1116, 226, 148, -1032, 618, -502, -1082, 704, 96, -166, 30, -70, -168, 24, 22600, 3588, -352,
                    2708, 0, -32468, 0, -14688, 4880, -12800, 16384, -8414, 16382, 9256, -2990, 29142, 32766, 0, 16384,
                    0, 0, 16382, 0, 0, -100, 0, 0, -186, 66, 5548, -17844, -31942, -29886, 14632,
                ],
                [-738, 0, 4028, 0, 4096, 0, -4029, 0, -738, -927, -503, 451],
                [
                    [-869, -454, 484],
                    [-920, -263, 491],
                    [-837, -85, 479],
                    [-885, -454, 397],
                    [-927, -264, 429],
                    [-845, -85, 406],
                ],
            ),
            // vsync 2582, frame 16
            (
                [
                    144, -1120, -456, -272, -1028, -142, -520, -1352, -736, 128, -258, -520, -374, -884, -408, 29674,
                    13790, 3940, 3334, 0, 0, 0, 30876, -5480, -27150, -21330, 9084, 27054, -2774, 6262, 31046, -32768,
                    0, 16384, 4420, 318, 2308, 0, 96, -8, 0, -90, 0, 4106, 7328, -17870, -4690, -2778, -12048,
                ],
                [694, 0, -4037, 0, 4096, 0, 4037, 0, 694, -609, -504, 469],
                [
                    [-574, -439, 430],
                    [-685, -280, 390],
                    [-584, -125, 456],
                    [-551, -523, 411],
                    [-747, -517, 386],
                    [-669, -402, 245],
                ],
            ),
        ];
        for (c, s0, legs) in rows {
            let s0 = s0.map(|v| v as f64);
            let pose = sk.pose(&c, None)?;
            let at =
                |s: usize| MESH_SLOT.iter().find(|x| x.1 == s).and_then(|x| pose[x.0]).context("a slot is not posed");
            let (r0, t0) = at(0)?;
            let m0 =
                [[s0[0], s0[1], s0[2]], [s0[3], s0[4], s0[5]], [s0[6], s0[7], s0[8]]].map(|r| r.map(|v| v / 4096.0));
            for (k, p) in legs.iter().enumerate() {
                let ours = mv(&r0, true, &sub(&at(12 + k)?.1, &t0));
                let ps1 = mv(&m0, true, &sub(&p.map(|v| v as f64), &[s0[9], s0[10], s0[11]]));
                assert!(np::norm(&sub(&ours, &ps1)) < 5.0, "slot {}: {ours:?}, the PS1's {ps1:?}", 12 + k);
            }
        }
        Ok(())
    }

    /// sin(ang(-5898)), the same on every platform (Python's math.sin on macOS gives one bit more)
    #[test]
    fn sin_golden() {
        let v = std::hint::black_box(-5898.0);
        assert_eq!(rot_zyx(0.0, v, 0.0)[2][0].to_bits(), 0x3fe125557dd41c96);
    }
}
