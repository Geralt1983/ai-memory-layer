// Tiny unified diff applier for single-file patches.
// For multi-hunk patches, we apply hunks sequentially against the current buffer.
// If any hunk fails to match, returns null.

export function applyUnifiedPatch(original: string, unified: string, filePath: string): string | null {
  const lines = unified.split(/\r?\n/)
  // Find the section that corresponds to this file (best-effort)
  // We accept patches that include only the hunks too.
  let idx = 0
  // Skip header lines until first @@
  while (idx < lines.length && !lines[idx].startsWith('@@')) idx++
  let output = original.split('\n')
  if (idx >= lines.length) return null

  while (idx < lines.length) {
    if (!lines[idx].startsWith('@@')) { idx++; continue }
    const m = lines[idx].match(/^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/)
    if (!m) return null
    const oldStart = parseInt(m[1], 10)
    const oldLen = m[2] ? parseInt(m[2], 10) : 1
    const newStart = parseInt(m[3], 10)
    // const newLen = m[4] ? parseInt(m[4], 10) : 1
    idx++

    // Collect hunk lines until next @@ or end
    const hunk: string[] = []
    while (idx < lines.length && !lines[idx].startsWith('@@')) {
      hunk.push(lines[idx])
      idx++
    }

    // Apply hunk
    // Convert to zero-based indices
    const base = oldStart - 1
    let oPtr = base
    const newChunk: string[] = []
    let oIdx = base

    // Build expected old segment to validate
    const oldSegment: string[] = []
    for (const l of hunk) {
      if (l.startsWith('-') || l.startsWith(' ')) {
        oldSegment.push(l.slice(1))
      }
    }
    const actualOld = output.slice(base, base + oldLen).join('\n')
    if (oldSegment.join('\n') !== actualOld) {
      return null
    }

    // Now build the replacement segment
    for (const l of hunk) {
      const tag = l[0]
      const text = l.slice(1)
      if (tag === ' ') { newChunk.push(text); oIdx++ }
      else if (tag === '-') { oIdx++ }
      else if (tag === '+') { newChunk.push(text) }
      else if (l.trim() === '') { newChunk.push('') }
    }

    // Splice into output
    output.splice(base, oldLen, ...newChunk)
  }

  // Normalize final newline behavior to match common git text files
  return output.join('\n')
}