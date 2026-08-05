from mmdet.core import EvalHook
import torch
import numpy as np

class SegEvalHook(EvalHook):
    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super().__init__(*args, dynamic_intervals, **kwargs)
        self.rule = 'greater'
        self.compare_func = self.rule_map[self.rule]
        self.best_ckpt_path = None
        self.key_indicator = 'Dice'

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return
        
        runner.model.eval()
        
        seg_results = []
        for data in self.dataloader:
            batch_data = {"img": [data["image"]],
                    "img_metas": [[data["img_metas"]]]}

            with torch.no_grad():
                seg_pred = runner.model(return_loss=False, rescale=True, **batch_data)

            seg_pred = seg_pred.squeeze().cpu().numpy()
            seg_ref = data["label"].squeeze().cpu().numpy()
            seg_result = compute_metrics(seg_ref, seg_pred, 
                                         self.dataloader.dataset.num_classes)
            seg_results.append(seg_result)
        
        # mean metric per class
        means = {}
        for r in range(1, self.dataloader.dataset.num_classes+1):
            means[r] = {}
            means[r]['Dice'] = np.nanmean([seg_result['metrics'][r]['Dice'] for seg_result in seg_results])

        # foreground mean
        foreground_mean = {}
        values = []
        for k in means.keys():
            values.append(means[k]['Dice'])
        foreground_mean['Dice'] = np.nanmean(values)

        self.latest_results = seg_results
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = foreground_mean['Dice']

        if not np.isnan(key_score):
            self._save_ckpt(runner, key_score)




def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn



def compute_metrics(seg_ref, seg_pred, num_classes) -> dict:
    
    results = {}
    results['metrics'] = {}
    for r in range(1, num_classes+1):
        results['metrics'][r] = {}
        mask_ref = seg_ref == r
        mask_pred = seg_pred == r
        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred)
        if tp + fp + fn == 0:
            results['metrics'][r]['Dice'] = np.nan
        else:
            results['metrics'][r]['Dice'] = 2 * tp / (2 * tp + fp + fn)
        
    return results