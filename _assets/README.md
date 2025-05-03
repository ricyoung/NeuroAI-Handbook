# NeuroAI Handbook Assets

This directory contains custom assets for the NeuroAI Handbook.

## Contents

### Cover Page

To use a custom cover page, add a PDF file named `cover.pdf` to this directory. The build scripts will automatically use this file instead of generating a default cover.

### Tagline

The default tagline used in the generated cover is:

> "Where Neurons Meet Algorithms: Bridging the Gap Between Brain Science and AI"

## Usage

The build scripts check this directory first when looking for assets. If a custom asset is found, it will be used instead of generating a default one.

## Adding New Assets

You can add the following types of assets to customize the handbook:

1. `cover.pdf`: Custom cover page
2. `logo.png`: Logo for the handbook (if needed)
3. `background.png`: Custom background image (if needed)
4. Any other custom graphics or PDFs needed for the handbook

## Notes

- PDF files should be letter size (8.5" x 11")
- Images should be high resolution (300 DPI or higher)
- Vector graphics (SVG, PDF) are preferred for best quality