// HumanEgo Sim — three.js viewer: MuJoCo kinematic tree + recorded rollouts.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { SoftBread } from './softbread.js';

const TMP_M = new THREE.Matrix4(), TMP_M2 = new THREE.Matrix4();
const TMP_Q = new THREE.Quaternion(), TMP_Q2 = new THREE.Quaternion();
const TMP_V = new THREE.Vector3(), TMP_V2 = new THREE.Vector3();
const ONE = new THREE.Vector3(1, 1, 1);
const qw = (q) => new THREE.Quaternion(q[1], q[2], q[3], q[0]);          // MuJoCo wxyz -> three xyzw
const easeOutBack = (x) => { const c1 = 1.5, c3 = c1 + 1; return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2); };

export const CAMS = {
  front: { pos: [1.42, -0.18, 0.66], target: [0.36, 0.0, 0.08] },
  side: { pos: [0.48, -1.28, 0.52], target: [0.42, 0.0, 0.07] },
  top: { pos: [0.62, 0.0, 1.42], target: [0.44, 0.0, 0.0] },
  orbit: { pos: [1.30, 0.62, 0.70], target: [0.38, 0.0, 0.08] },
};

export class SimViewer {
  constructor(container, opts = {}) {
    this.container = container;
    this.loader = new GLTFLoader();
    this.texLoader = new THREE.TextureLoader();
    this.cache = new Map();
    this.overlays = { traj: true, kpts: false, frames: false };   // keypoints / object frames off by default
    this.time = 0;

    const r = this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.NeutralToneMapping;
    r.toneMappingExposure = 1.05;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.VSMShadowMap;
    r.setClearColor(0x000000, 0);
    container.appendChild(r.domElement);
    r.domElement.style.background = 'radial-gradient(120% 95% at 50% 38%, #d4d3cf 0%, #bdbcb8 46%, #9f9e9a 100%)';

    const scene = this.scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(r);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environmentIntensity = 0.55;

    const cam = this.camera = new THREE.PerspectiveCamera(36, 16 / 9, 0.02, 30);
    cam.up.set(0, 0, 1);
    this.controls = new OrbitControls(cam, r.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.minDistance = 0.25;
    this.controls.maxDistance = 4.0;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.addEventListener('start', () => { this.userInteracted = true; this.camTween = null; this.autoOrbit = false; this.onUserCamera?.(); });

    // ---- lights ----
    const hemi = new THREE.HemisphereLight(0xfaf7f0, 0x8a8781, 0.55);
    scene.add(hemi);
    const key = this.key = new THREE.DirectionalLight(0xfff6ea, 2.1);
    key.position.set(1.4, -1.0, 2.6);
    key.target.position.set(0.42, 0, 0);
    key.castShadow = true;
    key.shadow.mapSize.set(4096, 4096);
    const sc = key.shadow.camera; sc.left = -1.1; sc.right = 1.1; sc.top = 1.1; sc.bottom = -1.1; sc.near = 0.5; sc.far = 6;
    key.shadow.bias = -0.0004; key.shadow.normalBias = 0.0; key.shadow.radius = 9; key.shadow.blurSamples = 20;
    scene.add(key, key.target);
    const rim = new THREE.DirectionalLight(0xdfe8ff, 0.55);
    rim.position.set(-1.6, 1.8, 1.6);
    scene.add(rim);

    // ---- seamless studio floor (radial fade into the backdrop) + shadow catcher ----
    const fc = document.createElement('canvas'); fc.width = fc.height = 512;
    const g = fc.getContext('2d');
    const grd = g.createRadialGradient(256, 256, 40, 256, 256, 256);
    grd.addColorStop(0, 'rgba(255,255,255,1)'); grd.addColorStop(0.55, 'rgba(255,255,255,0.85)'); grd.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grd; g.fillRect(0, 0, 512, 512);
    const fade = new THREE.CanvasTexture(fc);
    this.floorZ = -0.76;
    const floor = new THREE.Mesh(new THREE.CircleGeometry(4.2, 96),
      new THREE.MeshStandardMaterial({ color: 0xbebdb9, roughness: 0.92, metalness: 0, transparent: true, alphaMap: fade, depthWrite: false }));
    floor.position.set(0.4, 0, this.floorZ);
    floor.renderOrder = -2;
    scene.add(floor);
    const catcher = new THREE.Mesh(new THREE.CircleGeometry(3.0, 96), new THREE.ShadowMaterial({ opacity: 0.22, transparent: true }));
    catcher.position.set(0.4, 0, this.floorZ + 0.0005);
    catcher.receiveShadow = true;
    catcher.renderOrder = -1;
    scene.add(catcher);

    this.root = new THREE.Group();
    scene.add(this.root);
    this.overlayRoot = new THREE.Group();
    scene.add(this.overlayRoot);

    this.hq = opts.hq ?? !/Mobi|Android/i.test(navigator.userAgent);
    if (this.hq) {
      this.composer = new EffectComposer(r);
      this.composer.addPass(new RenderPass(scene, this.camera));
      const ao = this.ao = new GTAOPass(scene, this.camera, 640, 360);
      ao.updateGtaoMaterial({ radius: 0.035, distanceExponent: 1.4, thickness: 0.6, scale: 1.0, samples: 16 });
      ao.blendIntensity = 0.7;
      this.composer.addPass(ao);
      this.composer.addPass(new OutputPass());
    }
    this.setView('front', true);
    this.resize();
    new ResizeObserver(() => this.resize()).observe(container);
    this.clock = new THREE.Clock();
    this.playing = false;
    this.speed = 1;
    this.autoOrbit = false;
    r.setAnimationLoop(() => this._tick());
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    if (this.composer) { this.composer.setSize(w, h); this.composer.setPixelRatio(this.renderer.getPixelRatio()); }
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    for (const l of [...(this.trajLines || []), ...(this.trajGhosts || [])]) l.material.resolution.set(w, h);
  }

  // ------------------------------------------------------------------ assets
  async _gltf(url) {
    if (!this.cache.has(url)) this.cache.set(url, this.loader.loadAsync(url));
    return this.cache.get(url);
  }
  _tex(url, mat) {
    const key = 'tex:' + url;
    if (!this.cache.has(key)) {
      const t = this.texLoader.load(url);
      t.colorSpace = THREE.SRGBColorSpace;
      t.anisotropy = 8;
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      const qf = new URLSearchParams(location.search).get('flipy');
      if (mat.flipY === false || qf === '0') t.flipY = false;
      if (mat.repeat) t.repeat.set(mat.repeat[0], mat.repeat[1]);
      this.cache.set(key, t);
    }
    return this.cache.get(key);
  }
  _material(model, name, rgba) {
    const key = `mat:${model.key}:${name}:${rgba}`;
    if (this.cache.has(key)) return this.cache.get(key);
    const spec = (name && model.materials[name]) || null;
    let m;
    if (spec) {
      const P = (spec.clearcoat || spec.coat) ? THREE.MeshPhysicalMaterial : THREE.MeshStandardMaterial;
      m = new P({ color: new THREE.Color(...spec.color), roughness: spec.roughness ?? 0.6, metalness: spec.metalness ?? 0 });
      if (spec.clearcoat) { m.clearcoat = spec.clearcoat; m.clearcoatRoughness = 0.08; }
      if (spec.map) m.map = this._tex(spec.map, spec);
      if (spec.lift) m.color.multiplyScalar(spec.lift);              // brighten very dark baked textures
      if (spec.clearcoat === undefined && spec.coat) { m.clearcoat = spec.coat; m.clearcoatRoughness = 0.35; }
      if (spec.envIntensity) m.envMapIntensity = spec.envIntensity;
    } else {
      // Menagerie materials are plain rgba: render them as painted / plastic surfaces
      const lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2];
      m = new THREE.MeshPhysicalMaterial({ color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
        roughness: lum > 0.6 ? 0.32 : 0.42, metalness: lum < 0.25 ? 0.25 : 0.0, clearcoat: lum > 0.6 ? 0.35 : 0.1,
        clearcoatRoughness: 0.25 });
    }
    this.cache.set(key, m);
    return m;
  }

  // packed robot meshes with flat shading ship without normals (web_pack.quantize_glb): rebuild them
  // once per geometry (the GTAO pass needs a normal attribute, so no material-side flatShading)
  _shaded(geo) {
    if (geo.attributes.normal) return geo;
    if (!geo.userData.flat) { const f = geo.index ? geo.toNonIndexed() : geo.clone(); f.computeVertexNormals(); geo.userData.flat = f; }
    return geo.userData.flat;
  }

  // ------------------------------------------------------------------ model
  async loadModel(model, onProgress) {
    if (this.model && this.model.key === model.key) return;
    this.root.clear();
    this.model = model;
    const bodies = model.bodies.map((b, i) => {
      const grp = new THREE.Group();
      grp.matrixAutoUpdate = false;
      grp.name = b.name;
      this.root.add(grp);
      return { ...b, grp, local: new THREE.Matrix4().compose(new THREE.Vector3(...b.pos), qw(b.quat), ONE),
               world: new THREE.Matrix4() };
    });
    this.bodies = bodies;
    this.objectBodies = {};
    const assets = [...new Set([...model.geoms.filter(g => g.asset).map(g => g.asset),
                                ...Object.values(model.soft || {}).map(sp => sp.asset)])];
    let done = 0;
    await Promise.all(assets.map(a => this._gltf(a).then(() => onProgress?.(++done / assets.length))));
    for (const g of model.geoms) {
      if (g.name === 'floor') continue;                       // the studio floor replaces MuJoCo's plane
      let obj;
      if (g.type === 'mesh') {
        const gl = await this._gltf(g.asset);
        if (g.node) {
          const src = gl.scene.getObjectByName(g.node);
          if (!src) { console.warn('missing node', g.node); continue; }
          const mesh = (src.isMesh ? src : src.children.find(c => c.isMesh)) || src;
          obj = new THREE.Mesh(this._shaded(mesh.geometry), this._material(model, g.material, g.rgba));
          // KHR_mesh_quantization (web_pack.quantize_glb): int16 positions, offset + scale on the node
          src.updateMatrix();
          if (mesh !== src) { mesh.updateMatrix(); obj.applyMatrix4(TMP_M.multiplyMatrices(src.matrix, mesh.matrix)); }
          else obj.applyMatrix4(src.matrix);
        } else {
          obj = gl.scene.clone(true);
          if (g.yup) obj.rotation.x = Math.PI / 2;
          // photoscans ship a baked AO; GTAO adds contact shading on top, so damp the baked one
          obj.traverse(o => { if (o.isMesh && o.material && o.material.aoMap) { o.material.aoMapIntensity = 0.3; o.material.envMapIntensity = 1.35; } });
          const ms = g.material && model.materials[g.material];
          if (ms) {   // e.g. the ceramic plate: physically based override, double sided
            // colours measured from photos are sRGB
            const mat = new THREE.MeshPhysicalMaterial({ color: new THREE.Color().setRGB(...ms.color, THREE.SRGBColorSpace),
              roughness: ms.roughness ?? 0.2, metalness: 0, clearcoat: ms.clearcoat ?? 0.8, clearcoatRoughness: 0.18,
              side: THREE.DoubleSide });
            obj.traverse(o => { if (o.isMesh) { o.material = mat; o.geometry.computeVertexNormals(); } });
          }
          const wrap = new THREE.Group(); wrap.add(obj); obj = wrap;
        }
      } else if (g.type === 'box') {
        obj = new THREE.Mesh(new THREE.BoxGeometry(2 * g.size[0], 2 * g.size[1], 2 * g.size[2]), this._material(model, g.material, g.rgba));
      } else if (g.type === 'cylinder') {
        obj = new THREE.Mesh(new THREE.CylinderGeometry(g.size[0], g.size[0], 2 * g.size[1], 48), this._material(model, g.material, g.rgba));
        obj.geometry.rotateX(Math.PI / 2);
      } else if (g.type === 'sphere') {
        obj = new THREE.Mesh(new THREE.SphereGeometry(g.size[0], 32, 16), this._material(model, g.material, g.rgba));
      } else continue;
      const holder = new THREE.Group();
      holder.position.set(...g.pos);
      holder.quaternion.copy(qw(g.quat));
      holder.add(obj);
      obj.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
      const b = bodies[g.body];
      b.grp.add(holder);
      if (b.name in model.objects) this.objectBodies[b.name] = b;
      if (b.name.includes('gripper')) { obj.userData.gripper = true; }
    }
    // soft breads: a deformable GLB in a group that carries the bread's rigid frame
    this.soft = {};
    for (const [name, sp] of Object.entries(model.soft || {})) {
      const gl = await this._gltf(sp.asset);
      const sb = new SoftBread(name, sp, gl.scene);
      this.root.add(sb.group);
      this.soft[name] = sb;
      this.objectBodies[name] = { name, grp: sb.group, soft: sb, joints: [] };
    }
    // gripper finger meshes get their own material instance so they can glow when latched
    this.fingerMats = {};
    for (const b of bodies) {
      const m = b.name.match(/^(right|left)\//);
      if (!m || !(model.finger_bodies || []).includes(b.name.slice(m[0].length))) continue;
      b.grp.traverse(o => {
        if (o.isMesh) {
          o.material = o.material.clone();
          o.material.emissive = new THREE.Color(0xc084fc);
          o.material.emissiveIntensity = 0;
          (this.fingerMats[m[1]] ||= []).push(o.material);
        }
      });
    }
    this._jointIndex();
    this._buildOverlays();
  }

  _jointIndex() {
    this.jlist = [];
    for (const b of this.bodies) for (const j of b.joints) this.jlist.push(j);
  }

  // forward kinematics over the MuJoCo tree from a qpos vector
  _fk(q) {
    const B = this.bodies;
    B[0].world.identity();
    for (let i = 1; i < B.length; i++) {
      const b = B[i];
      const free = b.joints.length && b.joints[0].type === 'free';
      if (free) {
        const a = b.joints[0].qadr;
        b.world.compose(TMP_V.set(q[a], q[a + 1], q[a + 2]), TMP_Q.set(q[a + 4], q[a + 5], q[a + 6], q[a + 3]), ONE);
      } else {
        b.world.multiplyMatrices(B[b.parent].world, b.local);
        for (const j of b.joints) {
          const v = q[j.qadr];
          if (j.type === 'hinge') {
            TMP_M.makeTranslation(j.pos[0], j.pos[1], j.pos[2]);
            TMP_M2.makeRotationAxis(TMP_V2.set(j.axis[0], j.axis[1], j.axis[2]), v);
            TMP_M.multiply(TMP_M2);
            TMP_M2.makeTranslation(-j.pos[0], -j.pos[1], -j.pos[2]);
            TMP_M.multiply(TMP_M2);
            b.world.multiply(TMP_M);
          } else if (j.type === 'slide') {
            b.world.multiply(TMP_M.makeTranslation(j.axis[0] * v, j.axis[1] * v, j.axis[2] * v));
          }
        }
      }
      b.grp.matrix.copy(b.world);
      b.grp.matrixWorldNeedsUpdate = true;
    }
  }

  // ------------------------------------------------------------------ rollout
  setRollout(ro) {
    this.ro = ro;
    this.frames = ro.frames;
    this.t0 = ro.frames[0][0];
    this.t1 = ro.frames[ro.frames.length - 1][0];
    this.qbuf = new Float64Array(this.model.nq);
    // frame columns: qpos[q_keep] then one pose (pos + wxyz) per soft bread (web_export.frame_layout)
    const L = ro.frames[0].length - 1 === this.model.nq ? null : this.model.frame_layout;
    this.qKeep = L ? L.q_keep : null;
    this.softCols = {};
    if (L) L.soft.forEach((n, i) => { this.softCols[n] = L.q_keep.length + 7 * i; });
    this.vbuf = new Float64Array(ro.frames[0].length - 1);
    this.quatCols = [];
    for (const j of this.jlist) if (j.type === 'free') {
      const c = this.qKeep ? this.qKeep.indexOf(j.qadr + 3) : j.qadr + 3;
      if (c >= 0) this.quatCols.push(c);
    }
    for (const c of Object.values(this.softCols)) this.quatCols.push(c + 3);
    this.deform = ro.deform || {};
    this.dbuf = {};
    this.polTimes = ro.policy.map(p => p.t);
    this.appear = ro.appear || {};
    this._buildKpts();
    this.lastPolIdx = -2;
    this.setTime(this.t0);
  }

  _interp(t) {
    const F = this.frames;
    let lo = 0, hi = F.length - 1;
    if (t <= F[0][0]) { hi = 0; } else if (t >= F[hi][0]) { lo = hi; } else {
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (F[mid][0] <= t) lo = mid; else hi = mid; }
    }
    const a = F[lo], b = F[hi];
    const s = hi === lo ? 0 : (t - a[0]) / (b[0] - a[0]);
    const v = this.vbuf;
    for (let i = 0; i < v.length; i++) v[i] = a[i + 1] + (b[i + 1] - a[i + 1]) * s;
    for (const c of this.quatCols) {                    // slerp free-joint / soft-bread quaternions (wxyz)
      const k = c + 1;                                  // +1 for the time column
      TMP_Q.set(a[k + 1], a[k + 2], a[k + 3], a[k]);
      TMP_Q2.set(b[k + 1], b[k + 2], b[k + 3], b[k]);
      TMP_Q.slerp(TMP_Q2, s);
      v[c] = TMP_Q.w; v[c + 1] = TMP_Q.x; v[c + 2] = TMP_Q.y; v[c + 3] = TMP_Q.z;
    }
    if (!this.qKeep) return v;
    const q = this.qbuf, K = this.qKeep;
    for (let i = 0; i < K.length; i++) q[K[i]] = v[i];
    return q;
  }

  // soft-bread residuals at time t (sparse track, linear in time; absent = undeformed)
  _deformAt(name, t) {
    const tr = this.deform[name];
    if (!tr || !tr.t.length || t < tr.t[0] || t > tr.t[tr.t.length - 1]) return null;
    const T = tr.t;
    let lo = 0, hi = T.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (T[mid] <= t) lo = mid; else hi = mid; }
    const ra = tr.r[lo], rb = tr.r[hi];
    if (!ra && !rb) return null;
    const s = hi === lo ? 0 : (t - T[lo]) / (T[hi] - T[lo]);
    const n = (ra || rb).length;
    const D = (this.dbuf[name] ||= new Float32Array(n));
    for (let i = 0; i < n; i++) D[i] = 1e-4 * ((ra ? ra[i] : 0) * (1 - s) + (rb ? rb[i] : 0) * s);
    return D;
  }

  _applySoftReplay(t) {
    const v = this.vbuf;
    for (const [name, c] of Object.entries(this.softCols)) {
      const sb = this.soft[name]; if (!sb) continue;
      sb.setPose([v[c], v[c + 1], v[c + 2]], [v[c + 3], v[c + 4], v[c + 5], v[c + 6]]);
      sb.setDisplacement(this._deformAt(name, t));
    }
  }

  _applySoftLive() {
    for (const sb of Object.values(this.soft || {})) {
      const P = (sb._P ||= new Float32Array(3 * sb.nNode));
      sb.nodes.forEach((bid, i) => { if (bid < 0) return; const e = this.bodies[bid].world.elements; P[3 * i] = e[12]; P[3 * i + 1] = e[13]; P[3 * i + 2] = e[14]; });
      sb.fromNodes(P);
    }
  }

  /** World matrix of an object's frame (soft breads: fitted frame of the deformation grid). */
  objectMatrix(name) { const b = this.objectBodies[name]; return b ? b.grp.matrix : null; }

  setTime(t) {
    if (!this.frames) return;
    this.time = Math.min(Math.max(t, this.t0), this.t1);
    this._fk(this._interp(this.time));
    this._applySoftReplay(this.time);
    // intro: objects pop in, then fall under the recorded physics
    for (const [name, b] of Object.entries(this.objectBodies)) {
      const ta = this.appear[name];
      const k = ta === undefined ? 1 : Math.min(Math.max((this.time - ta) / 0.18, 0), 1);
      const s = k <= 0 ? 0.0001 : easeOutBack(k);
      b.grp.visible = k > 0;
      if (k < 1) b.grp.matrix.scale(TMP_V.set(s, s, s));
    }
    this._updateOverlays();
    this.onTime?.(this.time);
  }

  // ------------------------------------------------------------------ overlays
  _buildOverlays() {
    this.overlayRoot.clear();
    this.trajLines = [];
    this.trajDots = [];
    const w = this.container.clientWidth, h = this.container.clientHeight;
    for (const side of this.model.sides) {
      const geo = new LineGeometry();
      geo.setPositions(new Array(150).fill(0));
      const mat = new LineMaterial({ linewidth: 3.2, vertexColors: true, transparent: true, opacity: 0.95, depthTest: true,
                                     worldUnits: false, dashed: false });
      mat.resolution.set(w, h);
      const line = new Line2(geo, mat);
      line.visible = false;
      line.renderOrder = 10;
      this.overlayRoot.add(line);
      this.trajLines.push(line);
      const gmat = new LineMaterial({ linewidth: 2.0, vertexColors: true, transparent: true, opacity: 0.28, depthTest: false,
                                      depthWrite: false, worldUnits: false });
      gmat.resolution.set(w, h);
      const ghost = new Line2(geo, gmat);
      ghost.renderOrder = 9; ghost.visible = false;
      this.overlayRoot.add(ghost);
      line.userData.ghost = ghost;
      (this.trajGhosts ||= []).push(ghost);
      const dot = new THREE.Mesh(new THREE.SphereGeometry(0.009, 20, 12),
        new THREE.MeshBasicMaterial({ color: 0xffd36b, transparent: true, opacity: 0.95, depthTest: false }));
      dot.renderOrder = 11;
      dot.visible = false;
      this.overlayRoot.add(dot);
      this.trajDots.push(dot);
    }
  }

  _buildKpts() {
    // keypoints / PCA frames attached to the object bodies (they ride along with the physics)
    for (const b of Object.values(this.objectBodies)) {
      for (const c of [...b.grp.children]) if (c.userData.kpts) b.grp.remove(c);
    }
    const K = this.ro.kpts || {};
    for (const [name, spec] of Object.entries(K)) {
      const b = this.objectBodies[name];
      if (!b) continue;
      const grp = new THREE.Group(); grp.userData.kpts = true;
      const color = new THREE.Color(spec.color);
      const geo = new THREE.SphereGeometry(0.0042, 12, 8);
      const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.95, depthTest: false });
      const inst = new THREE.InstancedMesh(geo, mat, spec.pts.length);
      spec.pts.forEach((p, i) => inst.setMatrixAt(i, TMP_M.makeTranslation(p[0], p[1], p[2])));
      inst.renderOrder = 12;
      grp.add(inst);
      if (spec.frame) {
        const ax = new THREE.AxesHelper(0.06);
        ax.matrixAutoUpdate = false;
        const f = spec.frame;   // 4x4 row-major, object PCA frame in the body frame
        ax.matrix.set(...f);
        ax.material.depthTest = false; ax.renderOrder = 13;
        ax.userData.frame = true;
        grp.add(ax);
      }
      b.grp.add(grp);
    }
    this._applyOverlayVis();
  }

  setOverlay(name, on) { this.overlays[name] = on; this._applyOverlayVis(); this._updateOverlays(true); }
  _applyOverlayVis() {
    for (const b of Object.values(this.objectBodies || {})) {
      for (const c of b.grp.children) if (c.userData.kpts) {
        c.visible = this.overlays.kpts || this.overlays.frames;
        for (const cc of c.children) cc.visible = cc.userData.frame ? this.overlays.frames : this.overlays.kpts;
      }
    }
  }

  _policyIdx(t) {
    const T = this.polTimes; if (!T || !T.length || t < T[0]) return -1;
    let lo = 0, hi = T.length - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (T[mid] <= t) lo = mid; else hi = mid - 1; }
    return lo;
  }

  _updateOverlays(force) {
    if (!this.ro) return;
    const t = this.time;
    // a prediction becomes visible when it is delivered (one policy period after its observation)
    const P = this.ro.policy_period || 0.3;
    const k = this._policyIdx(t - P);
    const since = k >= 0 ? (t - P - this.polTimes[k]) : 0;
    this.model.sides.forEach((side, si) => {
      const line = this.trajLines[si], dot = this.trajDots[si];
      const e = k >= 0 ? this.ro.policy[k].sides[side] : null;
      const show = this.overlays.traj && e && e.traj && e.traj.length > 1;
      line.visible = !!show; dot.visible = !!show; line.userData.ghost.visible = !!show;
      if (show && (force || k !== this.lastPolIdx)) {
        const pos = [], col = [];
        const c0 = new THREE.Color(0xff6a00), c1 = new THREE.Color(0xffd36b);
        e.traj.forEach((p, i) => { pos.push(p[0], p[1], p[2]); const c = c0.clone().lerp(c1, i / (e.traj.length - 1)); col.push(c.r, c.g, c.b); });
        line.geometry.setPositions(pos);
        line.geometry.setColors(col);
        line.computeLineDistances();
        const la = Math.min(Math.round((this.ro.look_ahead ?? 45) / (e.traj_step || 1)), e.traj.length - 1);
        dot.position.set(...e.traj[la]);
      }
      if (show) {
        const a = Math.min(1, since / 0.12);
        line.material.opacity = 0.35 + 0.6 * a;
        dot.scale.setScalar(0.7 + 0.5 * Math.abs(Math.sin(t * 5)));
      }
      // finger glow when the policy considers the bread latched
      const latched = e ? !!e.grasping : false;
      for (const m of this.fingerMats[side] || []) m.emissiveIntensity += ((latched ? 0.22 : 0) - m.emissiveIntensity) * 0.25;
    });
    this.lastPolIdx = k;
    this.currentPolicy = k;
  }

  // ------------------------------------------------------------------ live mode
  beginLive() {
    this.playing = false; this.frames = null; this.live = true;
    if (this.ao) this.ao.enabled = false;            // leave the GPU to the policy during live runs
    this.ro = { kpts: {}, policy: [], policy_period: 0.3, look_ahead: 45 };
    this.appear = {};
    this._buildKpts();
    for (const b of Object.values(this.objectBodies)) b.grp.visible = true;
  }
  endLive() { this.live = false; if (this.ao) this.ao.enabled = true; }
  setLiveKpts(kpts) { this.ro.kpts = kpts; this._buildKpts(); }
  setLiveState(qpos, entries, t) {
    if (!this.live) return;
    this._fk(qpos);
    this._applySoftLive();
    for (const b of Object.values(this.objectBodies)) b.grp.visible = true;
    this.time = t;
    this.model.sides.forEach((side, si) => {
      const line = this.trajLines[si], dot = this.trajDots[si], e = entries[side];
      const show = this.overlays.traj && e && e.traj && e.traj.length > 1;
      line.visible = !!show; dot.visible = !!show; line.userData.ghost.visible = !!show;
      if (show && e !== line.userData.entry) {
        line.userData.entry = e;
        const pos = [], col = [], c0 = new THREE.Color(0xff6a00), c1 = new THREE.Color(0xffd36b);
        e.traj.forEach((p, i) => { pos.push(p[0], p[1], p[2]); const c = c0.clone().lerp(c1, i / (e.traj.length - 1)); col.push(c.r, c.g, c.b); });
        line.geometry.setPositions(pos); line.geometry.setColors(col); line.computeLineDistances();
        dot.position.set(...e.traj[Math.min(this.ro.look_ahead ?? 45, e.traj.length - 1)]);
      }
      if (show) dot.scale.setScalar(0.7 + 0.5 * Math.abs(Math.sin(performance.now() / 200)));
      for (const m of this.fingerMats[side] || []) m.emissiveIntensity += (((e && e.grasping) ? 0.22 : 0) - m.emissiveIntensity) * 0.25;
    });
    this.onLiveTime?.(t, entries);
  }

  // ------------------------------------------------------------------ camera
  setView(name, instant = false) {
    this.autoOrbit = name === 'orbit';
    if (name === 'policy' && this.model) {
      const pc = this.model.policy_camera;
      const T = pc.T_cam_in_world;        // 4x4 row-major, OpenCV optical frame
      const p = new THREE.Vector3(T[3], T[7], T[11]);
      const fwd = new THREE.Vector3(T[2], T[6], T[10]);
      const fovy = 2 * Math.atan(pc.height / 2 / pc.K[4]) * 180 / Math.PI;
      this.camTween = { from: this._camState(), to: { pos: p, target: p.clone().add(fwd.multiplyScalar(0.6)), fov: fovy },
                        t: instant ? 1 : 0 };
      return;
    }
    const c = CAMS[name] || CAMS.front;
    this.camTween = { from: this._camState(), to: { pos: new THREE.Vector3(...c.pos), target: new THREE.Vector3(...c.target), fov: 36 },
                      t: instant ? 1 : 0 };
    if (instant) this._applyTween(1);
  }
  setCustomView(pos, target) {
    this.autoOrbit = false;
    this.camTween = { from: this._camState(), to: { pos: new THREE.Vector3(...pos), target: new THREE.Vector3(...target), fov: 36 }, t: 1 };
    this._applyTween(1); this.camTween = null;
  }
  _camState() { return { pos: this.camera.position.clone(), target: this.controls.target.clone(), fov: this.camera.fov }; }
  _applyTween(a) {
    const tw = this.camTween; if (!tw) return;
    const e = a < 0.5 ? 4 * a * a * a : 1 - Math.pow(-2 * a + 2, 3) / 2;
    this.camera.position.lerpVectors(tw.from.pos, tw.to.pos, e);
    this.controls.target.lerpVectors(tw.from.target, tw.to.target, e);
    this.camera.fov = tw.from.fov + (tw.to.fov - tw.from.fov) * e;
    this.camera.updateProjectionMatrix();
  }

  // ------------------------------------------------------------------ loop
  _tick() {
    const dt = Math.min(this.clock.getDelta(), 0.1);
    if (this.camTween) {
      this.camTween.t = Math.min(1, this.camTween.t + dt / 1.1);
      this._applyTween(this.camTween.t);
      if (this.camTween.t >= 1) this.camTween = null;
    } else if (this.autoOrbit) {
      const tg = this.controls.target;
      const off = this.camera.position.clone().sub(tg);
      off.applyAxisAngle(new THREE.Vector3(0, 0, 1), dt * 0.12);
      this.camera.position.copy(tg).add(off);
    }
    if (this.playing && this.frames) {
      let t = this.time + dt * this.speed;
      if (t >= this.t1) { t = this.t1; this.playing = false; this.onEnded?.(); }
      this.setTime(t);
    }
    this.controls.update();
    if (this.composer) this.composer.render(); else this.renderer.render(this.scene, this.camera);
  }
}
