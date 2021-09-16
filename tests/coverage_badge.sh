#!/bin/bash

COV_PC="$( find -name "index.html" -a -path "*/coverage[!a-zA-Z]*" -exec grep -A5 -Fe "Coverage report:" -- "{}" + | sed -r -e ' \~^.*class="pc_cov">([^<>]+).*$~!d; s~~\1~; s~%~~; tk; d; :k  ' | sort -k1n | tail -1 )"
cov_grn=$(( COV_PC * 2550 / 1000)); cov_grn_hex="$( printf "%02x" $cov_grn; )"
cat <<EOF1
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="99" height="20">
    <linearGradient id="b" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <mask id="a">
        <rect width="99" height="20" rx="3" fill="#fff"/>
    </mask>
    <g mask="url(#a)">
        <path fill="#555" d="M0 0h63v20H0z"/>
        <path fill="#df${cov_grn_hex}25" d="M63 0h36v20H63z"/>
        <path fill="url(#b)" d="M0 0h99v20H0z"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
        <text x="31.5" y="15" fill="#010101" fill-opacity=".3">coverage</text>
        <text x="31.5" y="14">coverage</text>
        <text x="80" y="15" fill="#010101" fill-opacity=".3">${COV_PC}%</text>
        <text x="80" y="14">${COV_PC}%</text>
    </g>
</svg>
EOF1

