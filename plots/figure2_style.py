import json
from pathlib import Path

from lxml import etree as E

SVG = "http://www.w3.org/2000/svg"
NS = {"s": SVG}


def apply(source, config, dest):
    cfg = json.loads(Path(config).read_text("utf8"))
    tree = E.parse(source)
    root = tree.getroot()
    ids = {e.get("id"): e for e in root.iter() if e.get("id")}
    for key, attrs in cfg["groups"].items():
        for name, value in attrs.items():
            ids[key].set(name, value)
    for key, xml in cfg["texts"].items():
        old = ids[key]
        new = E.fromstring(xml.encode("utf8"))
        for e in new.iter():
            for k in list(e.attrib):
                if k.startswith("{") and (
                    not k.startswith("{http://www.w3.org/1999/xlink}")
                ):
                    del e.attrib[k]
        E.cleanup_namespaces(new)
        old.getparent().replace(old, new)
    for key in cfg["remove_decoration_groups"]:
        el = ids[key]
        el.getparent().remove(el)
    for k, v in cfg["canvas"].items():
        root.set(k, v)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(dest), encoding="utf-8", xml_declaration=True)
