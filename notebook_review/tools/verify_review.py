"""Verify clean notebook structure, paired exports, provenance and unchanged code ASTs."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def src(cell):
    return "".join(cell["source"])


def tree(text):
    return ast.dump(ast.parse(text), include_attributes=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--rf", type=Path, required=True)
    args = parser.parse_args()
    originals = {
        key: json.loads(path.read_text(encoding="utf-8"))["cells"]
        for key, path in (("yolo", args.yolo), ("rf", args.rf))
    }
    manifest = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    for key, path in (("yolo", args.yolo), ("rf", args.rf)):
        assert sha(path) == manifest["sources"][key]["sha256"], "Original notebook changed"
    covered = {"yolo": set(), "rf": set()}
    compared = set()
    code_cells = 0
    for name, entry in manifest["notebooks"].items():
        path = ROOT / (name + ".ipynb")
        py = ROOT / (name + ".py")
        assert sha(path) == entry["notebook_sha256"]
        assert sha(py) == entry["python_sha256"]
        nb = json.loads(path.read_text(encoding="utf-8"))
        assert (nb["nbformat"], nb["nbformat_minor"]) == (4, 5)
        assert len({c["id"] for c in nb["cells"]}) == len(nb["cells"])
        code = []
        grouped = {}
        for c in nb["cells"]:
            assert src(c).strip(), f"Empty cell: {name}"
            if c["cell_type"] != "code":
                continue
            code_cells += 1
            assert c["execution_count"] is None and c["outputs"] == []
            assert set(c["metadata"]) == {"review_source"}
            compile(src(c), f"{name}:{c['id']}", "exec")
            code.append(src(c))
            origin = c["metadata"]["review_source"]
            grouped.setdefault(origin, []).append(src(c))
            key, indices = origin.split(":")
            covered[key].update(map(int, indices.split("+")))
        assert tree("\n".join(code)) == tree(py.read_text(encoding="utf-8")), "Paired export differs"
        for origin, parts in grouped.items():
            key, index = origin.split(":")
            if "+" in index or origin == "yolo:1":
                continue  # Documented environment/configuration restructuring.
            index = int(index)
            actual = "\n".join(parts)
            expected = src(originals[key][index])
            if key == "yolo":
                if index in (6, 7):
                    nodes = ast.parse(actual).body
                    assert len(nodes) == 1 and isinstance(nodes[0], ast.If)
                    assert ast.dump(nodes[0].test) == ast.dump(ast.parse("not DO_RESUME", mode="eval").body)
                    assert ast.dump(ast.Module(body=nodes[0].body, type_ignores=[])) == tree(expected)
                    compared.add(origin)
                    continue
                if index == 9:
                    expected = expected[expected.index("if DO_RESUME:"):]
                if index == 11:
                    expected = expected.replace("GENERATE_SUBMISSION = True\n", "")
            if name.endswith("07_rfdetr_new_images") and index == 10:
                expected = expected.split("def box_iou(")[0]
            assert tree(actual) == tree(expected), f"Unexpected code change: {name}, {origin}"
            compared.add(origin)
    for key in covered:
        assert covered[key] == set(manifest["sources"][key]["nonempty_code_cells"]), "Missing source code cell"
    for worker, expected in manifest["embedded_workers"].items():
        path = ROOT / worker
        assert sha(path) == expected
        compile(path.read_text(encoding="utf-8"), worker, "exec")
    # Fresh training and resume must remain mutually exclusive, including Run all.
    main_nb = json.loads((ROOT / "01_yolo26l_training.ipynb").read_text(encoding="utf-8"))
    by_origin = {}
    for c in main_nb["cells"]:
        if c["cell_type"] == "code":
            by_origin.setdefault(c["metadata"]["review_source"], []).append(src(c))
    for mode, skipped in ((True, (6, 7)), (False, (9,))):
        for index in skipped:
            # Empty namespace makes any accidental execution of a skipped branch fail.
            exec("\n".join(by_origin[f"yolo:{index}"]), {"DO_RESUME": mode})
    result = {
        "status": "passed", "notebooks": len(manifest["notebooks"]),
        "code_cells_compiled": code_cells, "original_code_cells_accounted_for": sum(map(len, covered.values())),
        "source_groups_ast_checked": len(compared), "embedded_workers": len(manifest["embedded_workers"]),
        "paired_python_ast_matches": True, "original_hashes_unchanged": True,
        "fresh_resume_skip_checks": "passed", "gpu_training_executed": False,
        "framework_api_compatibility_tested": False,
    }
    (ROOT / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
