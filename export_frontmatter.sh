#!/bin/bash
# DEPRECATED: This script has been integrated into 02_build_supporting.py
# Please use python3 02_build_supporting.py instead

echo "WARNING: export_frontmatter.sh is deprecated and has been integrated into 02_build_supporting.py"
echo "Please use 'python3 02_build_supporting.py' instead."
echo "This script is kept for backward compatibility but may be removed in the future."

# Call the new integrated script to maintain backward compatibility
python3 $(dirname "$0")/02_build_supporting.py