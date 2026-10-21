// Live mode orchestrator: MuJoCo WASM physics + ONNX policy + three.js perception, driving the viewer.
import loadMujoco from 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.16.0/mujoco.js';
import * as M from './math.js';
import { LivePolicy } from './policy.js';
import { PolicyCamera, mirrorPose } from './perception.js';
import { LiveEpisode } from './runner.js';

let MJ = null;
async function mujoco() { if (!MJ) MJ = await loadMujoco(); return MJ; }

export class LiveSession {
  constructor(viewer, status) { this.viewer = viewer; this.status = status || (() => {}); this.running = false; this.speed = 1; }

  async load(key) {
    if (this.key === key) return;
    this.stop();
    const base = `assets/live/${key}/`;
    this.status('Loading MuJoCo (WebAssembly)…');
    const mj = await mujoco();
    const files = await (await fetch(base + 'files.json')).json();
    this.spec = await (await fetch(base + 'spec.json')).json();
    const root = `/live_${key}`;
    try { mj.FS.mkdir(root); mj.FS.mkdir(root + '/meshes'); } catch (e) { /* exists */ }
    this.status('Loading collision meshes…');
    await Promise.all([files.xml, ...files.files].map(async (f) => mj.FS.writeFile(`${root}/${f}`, new Uint8Array(await (await fetch(base + f)).arrayBuffer()))));
    this.model?.delete?.(); this.data?.delete?.();
    this.model = mj.MjModel.from_xml_path(`${root}/scene.xml`);
    this.data = new mj.MjData(this.model);
    if (!this.spec.scene_only && !this.policy) this.policy = await LivePolicy.create('assets/policy/', this.status);
    this.cam = this.spec.scene_only ? null : new PolicyCamera(this.viewer, this.viewer.model.policy_camera);
    this.key = key; this.mj = mj;
    this.status(this.spec.scene_only ? 'Ready · physics preview' :
      `Ready · ${this.policy.device === 'webgpu' ? 'WebGPU · 20 flow steps' : 'CPU · 10 flow steps (slower)'}`);
  }

  async run(scene, seed = 0, onPolicy, onDone) {
    this.stop();
    const ep = this.ep = new LiveEpisode({ mj: this.mj, model: this.model, data: this.data, spec: this.spec, policy: this.policy, cam: this.cam });
    ep.reset(scene, seed);
    ep.onPolicy = (side, e) => onPolicy?.(side, e, ep);
    ep.onDone = (res) => { this.running = false; onDone?.(res, ep); };
    this.viewer.beginLive(this.spec);
    this.running = true; this.paused = false;
    let simGoal = 0, last = performance.now(), busy = false, kptsDone = false;
    const loop = async () => {
      if (!this.running || this.ep !== ep) return;
      const now = performance.now(), dt = Math.min((now - last) / 1000, 0.1); last = now;
      if (!this.paused && !busy) {
        simGoal = Math.max(simGoal, ep.t) + dt * this.speed;
        busy = true;
        try { await ep.advanceTo(simGoal); } catch (e) { console.error(e); this.status('Live error: ' + e.message); this.running = false; }
        busy = false;
        if (!kptsDone && ep.objs) { kptsDone = true; this.viewer.setLiveKpts(this._kptsInBodies(ep)); }
      }
      this.viewer.setLiveState(this.data.qpos, ep.entries, ep.t - (ep.phase === 'settle' ? 0.6 : 0));
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  _kptsInBodies(ep) {
    // perception keypoints (camera frame) -> object body frames, so they ride with the physics in the viewer
    const out = {}, d = this.data, Tcw = this.spec.T_cam_in_world, colors = { obj1: '#22d3ee', obj2: '#3ddc97' };
    for (const [side, o] of Object.entries(ep.objs)) for (const [k, m] of Object.entries(o.meta)) {
      const body = k === 'obj1' ? `bread_${side}` : 'plate';
      if (out[body]) continue;
      let Ts = m.T_static, kl = m.kpts_local;
      if (side === 'left') { Ts = mirrorPose(Ts); kl = kl.map((p) => [-p[0], p[1], p[2]]); }
      let Tb;
      if (this.spec.objects[body].soft) {     // soft bread: the viewer's fitted frame of the deformation grid
        const e = this.viewer.objectMatrix(body).elements;   // column-major
        Tb = [e[0], e[4], e[8], e[12], e[1], e[5], e[9], e[13], e[2], e[6], e[10], e[14], 0, 0, 0, 1];
      } else {
        const b = this.spec.objects[body].body;
        const q = [d.xquat[4 * b + 1], d.xquat[4 * b + 2], d.xquat[4 * b + 3], d.xquat[4 * b]];
        Tb = M.fromRp(M.quatToMat(q), [d.xpos[3 * b], d.xpos[3 * b + 1], d.xpos[3 * b + 2]]);
      }
      const Tbi = M.inv4(Tb);
      out[body] = { pts: kl.map((p) => M.xform(Tbi, M.xform(Tcw, M.xform(Ts, p)))), color: colors[k],
                    frame: M.mul4(Tbi, M.mul4(Tcw, Ts)) };
    }
    return out;
  }

  pause(on) { this.paused = on; }
  stop() { this.running = false; this.ep = null; }
}
