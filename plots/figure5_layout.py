import re
from pathlib import Path

from lxml import etree as E

Q = lambda x: "{http://www.w3.org/2000/svg}" + x
PARSER = E.XMLParser(remove_blank_text=False, resolve_entities=False, no_network=True)


def compose(schematic, row, out):
    tree = E.parse(str(schematic), PARSER)
    root = tree.getroot()
    low = E.parse(row, PARSER).getroot()
    before = [E.tostring(x, method="c14n") for x in root.iter(Q("path"))]
    fig = low.find(Q("g"))
    index = {x.get("id"): x for x in fig}
    s = 0.9257922182109908
    ty = 262.1493180906538
    letters = {
        "".join(x.itertext()).strip(): x.get("id")
        for x in fig
        if x.tag == Q("g") and "".join(x.itertext()).strip() in ["b", "c", "d"]
    }
    letters = {k.strip(): v for k, v in letters.items()}
    placements = [
        ("b", ["axes_1", "axes_5", letters["b"]], -8.915162454873647),
        ("c", ["axes_2", letters["c"]], 5.6131167268351305),
        ("d", ["axes_3", letters["d"]], -9.02105896510227),
    ]
    for letter, names, tx in placements:
        g = E.Element(
            Q("g"),
            id="layout_panel_" + letter,
            transform=f"matrix({s:.12f},0,0,{s:.12f},{tx:.12f},{ty:.12f})",
        )
        for name in names:
            child = index[name]
            fig.remove(child)
            g.append(child)
        fig.append(g)
    for x in list(fig):
        if x.get("id") == "patch_1":
            fig.remove(x)
    for t in list(low.iter(Q("text"))):
        if t.xpath("string()") in ["b", "c", "d"]:
            t.getparent().remove(t)
    mapping = {
        x.get("id"): "aligned_quant_" + x.get("id") for x in low.iter() if x.get("id")
    }
    for x in low.iter():
        if not isinstance(x.tag, str):
            continue
        if x.get("id"):
            x.set("id", mapping[x.get("id")])
        for k, v in list(x.attrib.items()):
            v = re.sub(
                "url\\(#([^)]*)\\)",
                lambda m: "url(#" + mapping.get(m.group(1), m.group(1)) + ")",
                v,
            )
            if v.startswith("#") and v[1:] in mapping:
                v = "#" + mapping[v[1:]]
            x.set(k, v)
        if x.tag == Q("style") and x.text:
            x.text = x.text.replace("* {", "#fig5_aligned_quantitative * {")
    g = E.Element(Q("g"), id="fig5_aligned_quantitative")
    for child in list(low):
        if child.tag in [Q("defs"), Q("g")]:
            g.append(child)
    a = root.find(Q("text") + "[@id='figure5_panel_label_a']")
    assert a is not None
    root.insert(root.index(a), g)
    for letter, x in [("b", 1.5), ("c", 228.49759326113116), ("d", 430.4987966305656)]:
        t = E.SubElement(
            root,
            Q("text"),
            id="figure5_panel_label_" + letter,
            x=f"{x:.9f}",
            y=f"{276.44543022382703:.9f}",
        )
        t.text = letter
        t.set(
            "style",
            "font-family:Arial,Helvetica,sans-serif;font-weight:700;font-size:12px;text-anchor:start;fill:#1a1a1a",
        )
    assert before == [
        E.tostring(x, method="c14n")
        for x in root.iter(Q("path"))
        if not any(
            (p.get("id") == "fig5_aligned_quantitative" for p in x.iterancestors())
        )
    ]
    ids = [x.get("id") for x in root.iter() if x.get("id")]
    assert len(ids) == len(set(ids))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(
        E.tostring(tree, xml_declaration=True, encoding="UTF-8", pretty_print=False)
    )
