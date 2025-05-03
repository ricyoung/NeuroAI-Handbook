# NeuroAI Handbook Build Process

This document explains the build process for the NeuroAI Handbook PDF.

## Quick Start

To build the complete handbook with a single command:

```bash
python build_neuroai_handbook.py
```

This will generate the final PDF in the `pdf_exports/complete_handbook/` directory as `neuroai_handbook.pdf`.

## Custom Assets

To use custom assets like a custom cover page:

1. Place your custom `cover.pdf` file in the `_assets/` directory
2. Run the build script as usual
3. The build scripts will automatically detect and use your custom assets

The current tagline used in the generated cover is:
> "Where Neurons Meet Algorithms: Bridging the Gap Between Brain Science and AI"

## Build Process Components

The build system is organized into three main scripts:

1. `01_build_chapters.py`: Builds individual chapter PDFs
2. `02_build_supporting.py`: Creates supporting pages (cover, TOC, dividers)
3. `03_merge_final.py`: Combines everything into the final handbook

### 1. Building Chapters

The first script builds all individual chapter PDFs:

```bash
python 01_build_chapters.py
```

This processes all markdown files using JupyterBook and places the resulting PDFs in:
- `pdf_components/chapters/` (numbered with prefixes)
- `pdf_exports/chapters/` (without prefixes, for individual reference)

### 2. Creating Supporting Pages

The second script creates all supporting pages:

```bash
python 02_build_supporting.py
```

This creates:
- Cover page
- Copyright page
- Table of contents
- Part divider pages
- Acknowledgments page

These files are placed in:
- `pdf_components/supporting/`
- `pdf_exports/supporting/`

### 3. Merging the Final Handbook

The third script combines everything into the final handbook:

```bash
python 03_merge_final.py [options]
```

Options:
- `--output DIR`: Specify output directory (default: `pdf_exports/complete_handbook/`)
- `--basic`: Create only a basic version (no page numbers)
- `--acrobat-only`: Create only the Acrobat-compatible version

By default, this creates three versions of the handbook:
1. Complete version with page numbers (`neuroai_handbook.pdf`)
2. Acrobat-compatible version without page modifications (`neuroai_handbook_acrobat.pdf`)
3. Basic version for reference (`neuroai_handbook_basic.pdf`)

## Output Files

The build process creates the following directories:

- `pdf_components/`: Working files used during the build process
  - `chapters/`: Individual chapter PDFs with numbered prefixes
  - `supporting/`: Supporting page PDFs (cover, TOC, dividers)

- `pdf_exports/`: Final output files and individual components
  - `chapters/`: Individual chapter PDFs without prefixes
  - `supporting/`: Copies of supporting pages
  - `complete_handbook/`: Final combined handbook PDFs

## Troubleshooting

If you encounter issues:

1. Check the logs in the `logs/` directory for detailed information
2. Ensure that you've run the scripts in the correct order
3. Verify that all required files exist in the `pdf_components/` directory
4. Try running the build script with the `--basic` option for a simpler build

## Chapter Ordering

Note that Chapter 24 (Quantum Computing) is placed before Chapter 23 (Lifelong Learning) in the final handbook to maintain logical topic progression.

## Figure Quality

If figure quality issues are encountered:
- Check the SVG files in `book/figures/` directory
- Use the improved larger versions in the figures directory where available
- Modify the SVG files directly if text needs to be made more legible

## Complete Build Script

The `build_neuroai_handbook.py` script runs all three components in sequence:

```bash
python build_neuroai_handbook.py [--output DIR] [--acrobat-only] [--basic]
```

This is recommended for most use cases as it ensures all components are built consistently.