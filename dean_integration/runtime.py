"""Fail-closed AST hooks for the pinned entrypoint; no upstream files are edited."""
import ast


def source(node):
    return ast.unparse(node)


def exactly(tree, predicate, description):
    matches = [n for n in ast.walk(tree) if predicate(n)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one audited {description}, got {len(matches)}")
    return matches[0]


def patch_experiment(tree, *, online_only=False, use_selex_online=False, paired_online=False):
    if online_only and paired_online:
        raise ValueError("Strict replay and paired online are mutually exclusive")
    if online_only or paired_online:
        loop = exactly(tree, lambda n: isinstance(n, ast.For) and source(n.target) == "epoch"
                       and source(n.iter) == "range(0, args.nb_epochs)", "base loop")
        # Empty only the base loop; do not set nb_epochs=0 or alter official argv.
        loop.iter = ast.parse("range(0)", mode="eval").body
        load = exactly(tree, lambda n: isinstance(n, ast.Assign) and source(n) == "checkpoint = torch.load(pth_pth)", "base reload")
        load.value = ast.parse("_dean_load_base(pth_pth, model, header)", mode="eval").body
    if paired_online:
        loader = exactly(tree, lambda n: isinstance(n, ast.Assign) and len(n.targets) == 1
                         and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "dlod_tr_now",
                         "online stream DataLoader")
        if source(loader.value) != "torch.utils.data.DataLoader(copy.deepcopy(inc_tr_dataset), batch_size=args.inc_sz_batch, shuffle=True, num_workers=args.nb_workers)":
            raise ValueError("Audited online stream DataLoader changed")
        loader.value = ast.parse("_dean_paired.loader(copy.deepcopy(inc_tr_dataset), sess_idx, args.inc_sz_batch, args.nb_workers)", mode="eval").body
    if use_selex_online:
        loss = exactly(tree, lambda n: isinstance(n, ast.Assign) and source(n) ==
                       "loss = ce_loss + args.energy_hp * energy_loss", "online loss")
        original = loss.value
        loss.value = ast.parse("_dean_augment(_official_loss, feats, logits, on_y, header_now.head.weight, enabled=True, config=_dean_selex_config)", mode="eval").body
        loss.value.args[0] = original
    return tree


def instrument_boundary(tree, *, online_only=False, paired_online=False):
    if online_only and paired_online:
        raise ValueError("Strict replay and paired online are mutually exclusive")
    assignment = exactly(tree, lambda n: isinstance(n, ast.Assign) and len(n.targets) == 1
                         and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "pth_pth", "base checkpoint path")
    # Locate its actual containing block instead of using source line numbers.
    for parent in ast.walk(tree):
        for _, value in ast.iter_fields(parent):
            if isinstance(value, list) and any(n is assignment for n in value):
                index = next(i for i, n in enumerate(value) if n is assignment)
                if paired_online:
                    hook = "pth_pth = _dean_base_path\n_dean_paired.prepare(dt_train_inc_unlabelled, args.inc_sz_batch)"
                elif online_only:
                    hook = "pth_pth = _dean_base_path\n_dean_restore(_dean_boundary_rng)"
                else:
                    # Optional namespace hook keeps standalone AST audit fixtures executable.
                    hook = "if '_dean_save_boundary' in globals():\n    _dean_save_boundary(pth_pth, pth_rst_exp + '/base_boundary_rng.json', _dean_context)"
                value[index + 1:index + 1] = ast.parse(hook).body
                return tree
    raise ValueError("Missing base boundary block")
