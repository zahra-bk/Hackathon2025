import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(__file__))
DIST = os.path.join(ROOT, "dist")
PKG_DIR = os.path.join(ROOT, "pr_reviewer")
README = os.path.join(ROOT, "README.md")

os.makedirs(DIST, exist_ok=True)
zip_path = os.path.join(DIST, "pr_reviewer.zip")

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    # package
    for dirpath, _, filenames in os.walk(PKG_DIR):
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            arc = os.path.relpath(p, ROOT)
            z.write(p, arc)
    
    # demo directory if it exists
    demo_dir = os.path.join(ROOT, "demo")
    if os.path.exists(demo_dir):
        for dirpath, _, filenames in os.walk(demo_dir):
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                arc = os.path.relpath(p, ROOT)
                z.write(p, arc)
    
    # README
    if os.path.exists(README):
        z.write(README, "README.md")
    
    # CLI script
    cli_script = os.path.join(ROOT, "cli.py")
    if os.path.exists(cli_script):
        z.write(cli_script, "cli.py")

print(f"Created {zip_path}")
print(f"Size: {os.path.getsize(zip_path)} bytes")