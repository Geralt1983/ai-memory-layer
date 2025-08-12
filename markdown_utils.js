// Utilities for Markdown rendering with syntax highlighting
function renderMarkdown(text) {
    if (typeof marked !== 'undefined') {
        return marked.parse(text, {
            highlight: function(code, lang) {
                if (lang && hljs.getLanguage && hljs.getLanguage(lang)) {
                    return hljs.highlight(code, {language: lang}).value;
                }
                if (hljs.highlightAuto) {
                    return hljs.highlightAuto(code).value;
                }
                return code;
            },
            gfm: true,
            breaks: true
        });
    }
    return text;
}
