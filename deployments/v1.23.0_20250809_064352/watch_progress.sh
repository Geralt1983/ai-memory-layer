#!/bin/bash
echo "🚀 ChatGPT Import Live Monitor"
echo "==============================================="
echo "📊 Watching live progress..."
echo "Press Ctrl+C to exit (import continues running)"
echo "==============================================="
echo ""

# Function to show progress
show_progress() {
    if [ -f "./data/import_progress.json" ]; then
        echo "📈 Latest Progress:"
        cat ./data/import_progress.json | python3 -m json.tool 2>/dev/null || cat ./data/import_progress.json
        echo ""
    fi
}

# Show initial progress
show_progress

echo "📝 Live Activity (recent embedding operations):"
echo "-----------------------------------------------"

# Follow the log and filter for meaningful entries
tail -f import.log | grep --line-buffered -E "(Processing batch|Imported.*messages|embed_text.*text_length)" | while read line; do
    if [[ $line == *"text_length"* ]]; then
        # Extract text length from JSON log
        text_len=$(echo "$line" | grep -o '"text_length": [0-9]*' | cut -d' ' -f2)
        timestamp=$(echo "$line" | grep -o '"timestamp": "[^"]*"' | cut -d'"' -f4 | cut -d'T' -f2 | cut -d'.' -f1)
        echo "⚡ $timestamp - Processing text ($text_len chars)"
    elif [[ $line == *"Processing batch"* ]]; then
        echo "📦 $line"
        # Show updated progress when batch starts
        show_progress
    elif [[ $line == *"Imported"* ]] && [[ $line == *"messages"* ]]; then
        echo "✅ $line"
    fi
done