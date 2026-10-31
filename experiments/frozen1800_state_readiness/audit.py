"""Metadata-only admission audit. Never opens sealed arrays or runs a policy."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root, out = args.project, args.output
    out.mkdir(parents=True, exist_ok=True)
    evidence = []

    def read(rel):
        path = root / rel
        evidence.append({'path': rel, 'sha256': digest(path)})
        return json.loads(path.read_text())

    def save(name, data):
        (out / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

    raw = read('artifacts/action_prediction_evidence/AE500_robustness_and_census/data_census_v2/sealed_raw_replay_candidates.json')
    dataset = read('experiments/official_libero_rla_consistency/run_20261023_10/stove/dataset_selection.json')
    tz = read('artifacts/action_conditioned_wm/TZ_three_seed_500/independent_data/selection.json')
    held = read('artifacts/rla_policy_ood_and_data_readiness/held_out_manifest.json')
    dev = read('artifacts/rla_policy_ood_and_data_readiness/development_manifest.json')
    used = {x['source_demo']: x['split'] for x in dataset['episodes']}
    tz_used = {x['demo'] for x in tz['demonstrations'] if 'stove' in x['task']}
    rows = []
    for x in raw['demonstrations']:
        if 'stove' not in x['task']:
            continue
        demo = x['raw_demo']
        reasons = []
        if demo in used:
            reasons.append('current_' + used[demo])
        if demo in tz_used:
            reasons.append('prior_TZ_diagnostic_selection')
        rows.append({'registry': 'raw_demonstration', 'source_id': demo,
                     'identity': x['identity'], 'status': 'excluded_prior_use' if reasons else 'pending_full_use_and_alignment_audit',
                     'reason': ';'.join(reasons) or 'No overlap in checked lists; not yet certified independent',
                     'source_group': demo})
    dev_pairs = dev.get('development_clean_pairs', dev.get('clean_pairs', []))
    dev_keys = {(x['init_file'], x['initial_state_index']) for x in dev_pairs if 'stove' in x.get('base_task', '')}
    for x in held['held_out_clean_pairs']:
        if 'stove' not in x['base_task']:
            continue
        overlap = (x['init_file'], x['initial_state_index']) in dev_keys
        rows.append({'registry': 'official_init', 'source_id': str(x['initial_state_index']),
                     'identity': Path(x['init_file']).name + ':' + str(x['initial_state_index']),
                     'status': 'excluded_development_source_index' if overlap else 'unverified_source_demonstration',
                     'reason': 'Same init file/index in development; different reset seed does not establish independence' if overlap else 'Source demo mapping unavailable in registry; cannot certify exclusion from training or merge with raw candidates',
                     'source_group': 'unknown'})
    with open(out / 'candidate_inventory.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

    inv = root / 'experiments/rla_low_lr_continuation/run_20261030_01/delivery/checkpoint_inventory.csv'
    evidence.append({'path': str(inv.relative_to(root)), 'sha256': digest(inv)})
    models = []
    with open(inv) as f:
        for x in csv.DictReader(f):
            if int(x['step']) != 1800:
                continue
            path = Path(x['path'])
            actual = digest(path)
            models.append({**x, 'path': str(path.relative_to(root)), 'actual_sha256': actual,
                           'hash_matches': actual == x['sha256']})
            print('checkpoint', x['seed'], x['method'], 'hash_matches', actual == x['sha256'], flush=True)
    assert len(models) == 6 and all(x['hash_matches'] for x in models)
    save('checkpoint_inventory.json', models)
    save('input_metadata_hashes.json', evidence)
    summary = {
        'status': 'BLOCKED_INDEPENDENCE_ADMISSION',
        'raw_registered': sum(x['registry'] == 'raw_demonstration' for x in rows),
        'raw_excluded_prior_use': sum(x['status'] == 'excluded_prior_use' for x in rows),
        'raw_remaining_pending_not_certified': sum(x['status'] == 'pending_full_use_and_alignment_audit' for x in rows),
        'official_registered': sum(x['registry'] == 'official_init' for x in rows),
        'official_development_overlap': sum(x['status'] == 'excluded_development_source_index' for x in rows),
        'official_unknown_source': sum(x['status'] == 'unverified_source_demonstration' for x in rows),
        'certified_and_selected_N': 0,
        'note': 'These registries cannot establish 20 eligible independent sources. Counts from distinct registries cannot be added without a source crosswalk. This does not prove that no other eligible states exist.',
        'new_simulator_states_generated': 0,
        'sealed_payloads_read': False,
        'optimizer_updates': 0, 'AE_updates': 0, 'backward': 0,
        'policy_forward': 0, 'valid_rollouts': 0, 'attempted_rollouts': 0,
        'invalid_rollouts': 0, 'GPU_processes_started': 0,
        'checkpoint_hashes_pass': True,
        'history_protection': 'Metadata hashes verified unchanged after audit; checkpoints read-only; no historical result modified.',
    }
    assert all(digest(root / x['path']) == x['sha256'] for x in evidence)
    save('delivery_verification.json', summary)
    save('evaluation_readiness.json', {
        'status': 'NOT_FROZEN_FOR_SCORING', 'minimum_N': 20, 'maximum_N': 50,
        'selected_states': [], 'scoring_authorized_after_admission_only': True,
        'conditions': ['Clean', 'new_background_1', 'new_background_2', 'new_background_3', 'new_background_4'],
        'budget_formula': '30*N', 'max_valid_rollouts': 1500,
        'primary_estimand': 'Equal-background and equal-policy-seed mean success difference D-low minus C-low',
        'bootstrap_unit': 'Independent source demonstration, or physical state only when sources are independent',
        'stop_reason': 'Fewer than 20 sources certified by available metadata; do not substitute reused development states or silently pool unmapped registries.',
    })
    report = '''# 冻结 step1800 策略：独立状态准入报告

状态：BLOCKED（数据独立性准入），未开始正式封存评分。

上一轮提交 `442448c56c6eb075a895692033798395c36d8da8` 已通过仅对该命令禁用失效代理完成推送，并用远端 refs 核验。原失败记录保留。

## 清单结论

- 原始示范封存候选登记中，本任务有15条；5条后来进入当前训练/development，同时已出现在早期TZ诊断选择中，不能继续称为未使用。
- 剩余10条仅通过上述清单的初筛，尚未完成全部使用历史与时序验收；即使全合格也不足20条。更正此前进度消息中的“6条重叠”：准确交集为5条。
- 另一份官方初态登记有20个候选，其中3个（index 13、23、35）与development登记使用同一init文件及索引，不能因reset seed不同就视为独立来源。
- 其余17个缺少与来源示范的一一对应证据，无法排除训练来源重叠，也不能与上述10条相加凑出27条。
- 当前选定并认证状态数N=0。该结论表示现有登记不足以支持放行，并不证明所有可能数据源都不足。未读取封存状态数组、图像或轨迹内容。

按用户“不足20个合格状态则停止”的规则，本轮停在准入阶段。未以旧development补齐，未另行生成模拟器状态绕过该停止条件。

## 模型与历史保护

六个指定step1800文件均按上一轮inventory解析实际路径，重新计算SHA-256并匹配成功。未加载模型或改变权重。输入元数据结束时复核哈希未变。未修改历史报告或原失败推送记录。

## 能回答与不能回答

没有产生新的成功率；因此目前不能判断D-low的背景优势是否在独立状态上保留、是否存在Clean代价。历史development结果不改称独立验证。

唯一优先下一步：解决状态来源准入——补齐官方初态到源示范的可靠映射及使用记录，再判断能否形成至少20个独立来源；若无法形成，应明确另立、冻结模拟器生成状态方案，而非将不明来源登记为独立样本。本轮不自动执行替代方案。

optimizer updates=0，AE updates=0，backward=0，模型前向=0，有效闭环=0，实际尝试=0，无效闭环=0。未启动GPU进程，无本任务GPU资源需要释放。

交付含候选逐条排除/待核验原因、六模型哈希、只读元数据哈希、未放行协议草案和核验记录。未上传权重或原始数据。
'''
    (out / 'report.md').write_text(report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
