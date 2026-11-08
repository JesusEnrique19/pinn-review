# Float queue for wide figures, then PDF export (called by pinnreview.manuscript.export_pdf).
#
# Wide figures are Word frames pinned to the top of their anchor's page. Word does not stack frames, so the
# queue resolves collisions in document order. When a frame lands on the same page as the previous one, it is
# first stacked directly below that frame (absolute offset from the top margin) if both fit in $MaxShare of the
# text height; otherwise the block that follows it (a paragraph, or a caption with its table) is moved in front
# of it, so the frame drifts forward through the text to the next page. At most two frames share a page, frames
# never overtake one another, and a stacked frame whose anchor moves to another page goes back to the top and
# is not stacked again, so the "Fig. N" references stay valid and the loop terminates.
# The document is saved back (docx) and exported as PDF.
param([string]$Docx, [string]$Pdf, [double]$WidePoints, [double]$MaxShare = 0.72)

$ErrorActionPreference = 'Stop'
$wdFrameTop = -999999
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $doc = $word.Documents.Open($Docx)
    $ps = $doc.PageSetup
    $limit = $MaxShare * ($ps.PageHeight - $ps.TopMargin - $ps.BottomMargin)
    $gap = 12

    function Get-WideFrames {
        $out = @()
        foreach ($f in $doc.Frames) {
            if ([math]::Abs($f.Width - $WidePoints) -lt 2) { $out += $f }
        }
        $out | Sort-Object { $_.Range.Start }
    }

    function Get-FrameHeight($f) {
        # picture + space before + caption lines (8 pt captions, about 150 characters per full-width line)
        $h = 0
        foreach ($s in $f.Range.InlineShapes) { $h += $s.Height }
        $cap = $f.Range.Paragraphs.Last.Range.Text.Length
        return $h + 6 + 11 * [math]::Ceiling($cap / 150) + 6
    }

    function Get-Page($f) { $f.Range.Information(3) }

    $stacked = @{}
    $tried = @{}
    $filled = @{}
    $moves = 0
    $stacks = 0
    for ($iter = 0; $iter -lt 400; $iter++) {
        $doc.Repaginate()
        $frames = @(Get-WideFrames)
        # a stacked frame whose anchor left its partner's page goes back to the top
        $reset = $false
        foreach ($k in @($stacked.Keys)) {
            if ((Get-Page $frames[$k]) -ne (Get-Page $frames[$k - 1])) {
                $frames[$k].VerticalPosition = $wdFrameTop
                $stacked.Remove($k)
                $reset = $true
            }
        }
        if ($reset) { continue }
        $hit = -1
        for ($k = 1; $k -lt $frames.Count; $k++) {
            if ($stacked.ContainsKey($k)) { continue }
            if ((Get-Page $frames[$k]) -le (Get-Page $frames[$k - 1])) { $hit = $k; break }
        }
        $gapFill = $false
        if ($hit -lt 0) {
            # no collision left: a frame whose preceding text ends on an earlier page leaves that page short
            # (Word moves the anchor paragraph, and a heading before the frame is orphaned), so pull the next
            # block in front of the frame to fill the page
            for ($k = 0; $k -lt $frames.Count; $k++) {
                if ($stacked.ContainsKey($k) -or $frames[$k].Range.Start -lt 2) { continue }
                $before = $doc.Range($frames[$k].Range.Start - 1, $frames[$k].Range.Start - 1).Information(3)
                if ($before -lt (Get-Page $frames[$k]) -and $filled[$k] -lt 4) {
                    $filled[$k] = $filled[$k] + 1
                    $hit = $k
                    $gapFill = $true
                    break
                }
            }
        }
        if ($hit -lt 0) { break }
        $cur = $frames[$hit]
        $prev = if ($hit -gt 0) { $frames[$hit - 1] } else { $null }
        if (-not $gapFill -and -not $tried.ContainsKey($hit) -and -not $stacked.ContainsKey($hit - 1) -and
            (Get-Page $cur) -eq (Get-Page $prev)) {
            $hp = Get-FrameHeight $prev
            if ($hp + $gap + (Get-FrameHeight $cur) -le $limit) {
                $cur.RelativeVerticalPosition = 0  # margin
                $cur.VerticalPosition = $hp + $gap
                $stacked[$hit] = $true
                $tried[$hit] = $true
                $stacks++
                continue
            }
        }
        if (-not $gapFill) { $tried[$hit] = $true }
        # unit = this frame plus any frames directly after it; a frame's Range stops before its last
        # paragraph mark, and the unit must include it
        $start = $cur.Range.Start
        $end = $cur.Range.Paragraphs.Last.Range.End
        for ($k = $hit + 1; $k -lt $frames.Count; $k++) {
            if ($frames[$k].Range.Start -le $end) { $end = $frames[$k].Range.Paragraphs.Last.Range.End } else { break }
        }
        if ($end -ge $doc.Content.End - 1) { break }
        $here = $doc.Range($start, $start)
        $after = $doc.Range($end, $end)
        if ($after.Information(12)) {  # a table directly after the unit
            $tbl = $after.Tables(1)
            $here.FormattedText = $tbl.Range.FormattedText
            $tbl.Delete()
        } else {
            $para = $after.Paragraphs(1).Range
            $next = $doc.Range($para.End, $para.End)
            if (($para.End -lt $doc.Content.End - 1) -and $next.Information(12)) {
                # a caption directly above its table: move both (Word keeps a paragraph mark before a table)
                $tbl = $next.Tables(1)
                $here.FormattedText = $doc.Range($para.Start, $tbl.Range.End).FormattedText
                $tbl.Delete()
                $para.Delete() | Out-Null
            } else {
                $here.FormattedText = $para.FormattedText
                $para.Delete() | Out-Null
            }
        }
        $moves++
    }
    Write-Output "float queue: $moves move(s), $stacks stacked"
    $doc.SaveAs2($Docx, 16)
    $doc.SaveAs2($Pdf, 17)
    $doc.Close(0)
}
finally {
    $word.Quit()
}
