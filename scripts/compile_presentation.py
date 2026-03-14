import os
import subprocess
import shutil
import argparse

def compile_latex(tex_file, output_dir="presentation"):
    """Compiles a LaTeX file to PDF. Requires pdflatex to be installed."""
    
    if not os.path.exists(tex_file):
        print(f"Error: File {tex_file} not found.")
        return False

    abs_tex_path = os.path.abspath(tex_file)
    project_root = os.path.dirname(os.path.dirname(abs_tex_path))
    tex_dir = os.path.dirname(abs_tex_path)
    base_name = os.path.splitext(os.path.basename(tex_file))[0]
    
    # Paths to the generated figures
    figures = [
        "vis_coco_pretrain/coco_pretrain_sequence.png",
        "vis_ade_pretrain/ade_pretrain_sequence.png",
        "vis_ytvos/ytvos_sequence.png"
    ]
    
    # Ensure figures are accessible from the presentation directory
    for fig in figures:
        src = os.path.join(project_root, fig)
        dst = os.path.join(tex_dir, os.path.basename(fig))
        if os.path.exists(src):
            print(f"Linking/Copying figure: {fig} -> {dst}")
            if os.path.exists(dst): os.remove(dst)
            shutil.copy2(src, dst)
        else:
            print(f"Warning: Figure {src} not found. Compilation might fail.")

    print(f"Compiling {tex_file}...")
    
    try:
        # Run pdflatex twice for references/toc if needed
        for _ in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", os.path.basename(tex_file)],
                cwd=tex_dir,
                capture_output=True,
                text=True
            )
            
        if result.returncode == 0:
            print(f"Success! PDF generated: {os.path.join(tex_dir, base_name + '.pdf')}")
            return True
        else:
            print("Compilation failed.")
            print("--- STDOUT ---")
            print(result.stdout)
            print("--- STDERR ---")
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("Error: 'pdflatex' command not found. Please install TeX Live or a similar LaTeX distribution.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="presentation/pretraining_strategy.tex")
    args = parser.parse_args()
    
    compile_latex(args.file)
