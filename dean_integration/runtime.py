"""Fail-closed AST hooks for the pinned entrypoint; no upstream files are edited."""
import ast


def source(node):
    return ast.unparse(node)


def exactly(tree, predicate, description):
    matches = [n for n in ast.walk(tree) if predicate(n)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one audited {description}, got {len(matches)}")
    return matches[0]


def patch_experiment(tree, *, online_only=False, use_selex_online=False, paired_online=False, reference_selex=False):
    if use_selex_online and reference_selex:
        raise ValueError("Synthetic and reference SelEx hooks are mutually exclusive")
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
    if reference_selex:
        patch_real_selex(tree)
    return tree


def insert_after(tree, statement, text):
    for parent in ast.walk(tree):
        for _, value in ast.iter_fields(parent):
            if isinstance(value, list):
                for i, node in enumerate(value):
                    if node is statement:
                        value[i+1:i+1] = ast.parse(text).body
                        return
    raise ValueError("Audited insertion statement has no containing block")


def patch_real_selex(tree):
    session = exactly(tree, lambda n: isinstance(n, ast.Assign) and source(n) == "model_now = model_now.cuda()", "student CUDA initialization")
    expansion = exactly(tree, lambda n: isinstance(n, ast.Expr) and source(n) == "header_now.update(nb_classes_now=nb_classes_now)", "classifier expansion")
    loss = exactly(tree, lambda n: isinstance(n, ast.Assign) and source(n) == "loss = ce_loss + args.energy_hp * energy_loss", "online loss")
    groups = [n for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
              and isinstance(n.value.func, ast.Attribute) and source(n.value.func) == "opt.add_param_group"]
    if len(groups) != 2:
        raise ValueError("Expected both audited DEAN classifier optimizer additions")
    insert_after(tree, session, "_dean_real.start_session(model_now)")
    insert_after(tree, expansion, "_dean_real.prepare_batch(model_now, dset_tr_now_m, dataset.utils.make_transform, nb_classes_now)")
    for statement in groups:
        insert_after(tree, statement, "_dean_real.attach_optimizer(opt, header_now)")
    original = loss.value
    loss.value = ast.parse("_dean_real.augment(_original, ce_loss, args.energy_hp * energy_loss, model_now, on_y, on_z)", mode="eval").body
    loss.value.args[0] = original


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


def patch_bounded_smoke(tree):
    """Pinned smoke-only changes: suppress full base eval, terminate post-step."""
    session = exactly(tree, lambda n: isinstance(n, ast.Assign) and source(n) == "model_now = model_now.cuda()", "smoke student")
    insert_after(tree, session, "_dean_real.configure_replay_from_cub(dt_train_labelled, dataset.utils.make_transform)")
    loop = exactly(tree, lambda n: isinstance(n, ast.For) and source(n.iter) == "range(0, args.inc_epochs)", "smoke inner epochs")
    step = exactly(loop, lambda n: isinstance(n, ast.Expr) and source(n) == "opt.step()", "smoke optimizer step")
    insert_after(tree, step, "_dean_smoke.step_completed()")
    forward = exactly(loop, lambda n: isinstance(n, ast.Assign) and source(n) == "feats = model_now(on_x)", "smoke forward")
    index = loop.body.index(forward)
    loop.body[index:index] = ast.parse("_dean_smoke.update_started()").body
    main = exactly(tree, lambda n: isinstance(n, ast.If) and source(n.test) == "__name__ == '__main__'", "main block")
    marker = next((i for i, n in enumerate(main.body) if isinstance(n, ast.Expr) and source(n) == "print('==> Init. Evaluation..')"), None)
    if marker is None or not isinstance(main.body[marker+1], ast.With):
        raise ValueError("Audited smoke initial evaluation boundary changed")
    main.body[marker+1:marker+2] = ast.parse("acc_0 = None").body
    message = exactly(tree, lambda n: isinstance(n, ast.Expr) and "[Base Session] After Pretrain" in source(n), "base accuracy print")
    message.value = ast.parse("print('Bounded smoke: base accuracy not measured')", mode="eval").body
    return tree
