"""Physics-informed, geometry-preserving augmentation for simulated B-scans (boxes are NOT changed).
Each term models a measured sim->real gap (see results/REAL_RESULTS.md R-A7, R-A9):
  additive noise (sigma 12 cost 0.13-0.20 mAP50), contrast/gain (x0.5, x2 cost up to 0.05), global brightness/polarity (field images differ
  strongly in global statistics: pixel-std alone gives AUROC 0.94), vertical blur (finite antenna bandwidth), horizontal banding (system ringing /
  coupling), row-median background removal (standard field processing), and resolution loss (field images are small and rescaled).
Install with physaug.patch() BEFORE model.train(); applied only to training images, before ultralytics' Format step."""
import numpy as np, cv2
class PhysAug:
    def __init__(self, p_noise=0.7, sig=(2.0, 20.0), p_gain=0.6, gain=(0.5, 2.0), p_off=0.5, off=40.0, p_blur=0.3, blur=(0.5, 1.5),
                 p_band=0.3, band=(2.0, 12.0), p_inv=0.3, p_bg=0.3, p_res=0.5, res=(1.0, 3.2)):
        self.__dict__.update(locals()); del self.__dict__["self"]
    def __call__(self, labels):
        r = np.random; g = labels["img"][..., 0].astype(np.float32); mu = float(g.mean()); H, W = g.shape
        if r.rand() < self.p_res:                                   # resolution loss: down then bilinear up (field sets are small and rescaled)
            f = r.uniform(*self.res)
            if f > 1.05: g = cv2.resize(cv2.resize(g, (max(int(W / f), 8), max(int(H / f), 8)), interpolation=cv2.INTER_AREA), (W, H), interpolation=cv2.INTER_LINEAR)
        if r.rand() < self.p_bg: g = g - np.median(g, axis=1, keepdims=True) + mu          # row-median background removal
        if r.rand() < self.p_blur: g = cv2.GaussianBlur(g, (0, 0), sigmaX=0.01, sigmaY=r.uniform(*self.blur))   # vertical blur only
        if r.rand() < self.p_inv: g = 2 * mu - g                                             # polarity convention
        if r.rand() < self.p_gain: g = (g - mu) * float(np.exp(r.uniform(np.log(self.gain[0]), np.log(self.gain[1])))) + mu
        if r.rand() < self.p_band:                                                           # horizontal banding (smooth random row profile)
            prof = cv2.GaussianBlur(r.normal(0, 1, (H, 1)).astype(np.float32), (0, 0), sigmaX=0.01, sigmaY=r.uniform(1.0, 6.0))
            g = g + prof / (prof.std() + 1e-6) * r.uniform(*self.band)
        if r.rand() < self.p_off: g = g + r.uniform(-self.off, self.off)
        if r.rand() < self.p_noise: g = g + r.normal(0, r.uniform(*self.sig), g.shape).astype(np.float32)
        g = np.clip(g, 0, 255).astype(np.uint8); labels["img"] = np.repeat(g[..., None], 3, axis=2); return labels
def patch(**kw):
    from ultralytics.data.dataset import YOLODataset
    orig = YOLODataset.build_transforms
    def build(self, hyp=None):
        tr = orig(self, hyp)
        if self.augment: tr.transforms.insert(len(tr.transforms) - 1, PhysAug(**kw))
        return tr
    YOLODataset.build_transforms = build
if __name__ == "__main__":      # visual/self test
    import sys; from PIL import Image
    im = np.asarray(Image.open(sys.argv[1]).convert("L")); lab = {"img": np.repeat(im[..., None], 3, 2)}; np.random.seed(0)
    out = [np.asarray(PhysAug(p_noise=1, p_gain=1, p_off=1, p_blur=1, p_band=1, p_inv=0, p_bg=1, p_res=1)(dict(img=lab["img"].copy()))["img"][..., 0]) for _ in range(3)]
    Image.fromarray(np.hstack([im] + out)).save(sys.argv[2]); print("saved", sys.argv[2], out[0].shape, out[0].dtype)
