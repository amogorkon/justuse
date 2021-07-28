#!/bin/bash

[ -d use -a -f use/use.py ] && cd ..
[ -f unit_teat.py ] && cd ..
while [ ! -e requirements.txt ]; do
  cd ..
done

: ${PYTHON3:=python3}
export PYTHON3


if [ "x$1" != "xupload" ]; then
  typeset -p -x
  typeset -p
  typeset -p -a
  
  find . -regextype egrep "(" "(" -name .git -prune -false ")" -o -iregex '.*/\..*cache.*|.*cache.*/\..*' ")" -a -exec rm -vrf -- "{}" +
  rm -vrf -- ~/.justuse-python/packages
  rm  -vf -- ~/.justuse-python/registry.json
  
  yes y | $PYTHON3 -m pip install --force-reinstall --upgrade \
          -r requirements.txt
  which busybox 2>/dev/null || {
    if which apt; then
      apt install -y busybox || sudo apt install -y busybox
    fi
  }
fi


echo "=== COVERAGE SCRIPT ==="
echo "$0 running from $( pwd )"
cdir="$( pwd )"
cfile="$cdir/.coveragerc"
while [ ! -e "$cfile" ]; do
    cdir="${cfile%/?*}"
    cdir2="${cdir%/?*}"    
    [ "x${cfile2##/}" = "x/" ] && echo "No coveragerc" && exit 255
    cfile="$( find "$cdir2" -mindepth 2 -maxdepth 4 -name .coveragerc )"
done
echo "cfile=${cfile}" 1>&2
cdir="${cfile%/?*}"
cd "$cdir"

if [ "x$1" != "xupload" ]; then
  if [ ! -f .coveragerc ]; then
    echo "Cannot find .coveragerc after mpving to $cdir: $(pwd)"
    find "$( pwd )" -printf '%-12s %.10T@ %p\n' | sort -k2n
    exit 123
  fi
fi

if [ "x$1" != "xupload" ]; then
    if [ "x$GITHUB_AUTH" != "x" ]; then
        $PYTHON3 -m pip install --force-reinstall coverage \
            pytest-cov pytest-env
    fi
fi

set +e
default="/home/runner/work/justuse/justuse/coverage.svg"
file="${COVERAGE_IMAGE:-${default}}"
COVERAGE_IMAGE="$file"
export COVERAGE_IMAGE
f="$file"
fn="${f##*/}"
dir="${f: 0:${#f}-${#fn}}"; dir="${dir%%/}"
nme="${fn%.*}"

if [ "x$1" == "xupload" -a -f "$file" ]; then
    echo "Found an image to publish: [$file]" 1>&2
    for variant in \
        '"~/public_html/mixed/" "$file"'  \
        ' "/public_html/mixed" "$file"'  \
        '"~/public_html/mixed/" "$file"'  \
        ' "/public_html/mixed/" "$file"'  \
        '"~/public_html/mixed/$fn" "$file"'  \
        ' "/public_html/mixed/$fn" "$file"'  \
        '"~/public_html/mixed/$fn" "$file"'  \
        ' "/public_html/mixed/$fn" "$file"'  \
        '"/var/www/public_html/mixed/" "$file"'  \
        '"/var/www/public_html/mixed" "$file"'  \
        '"/var/www/public_html/mixed/$fn" "$file"'  \
        '"/var/www/public_html/mixed/$fn" "$file"'  \
        '"/var/www/mixed/" "$file"'  \
        '"/var/www/mixed" "$file"'  \
        '"/var/www/mixed/$fn" "$file"'  \
        '"~/mixed/$fn" "$file"'  \
        '"~public_html/mixed/$fn" "$file"'  \
        ;  \
    do
        eval "set -- $variant"
        cmd=(  busybox ftpput -v -P 21 -u "$FTP_USER" -p "$FTP_PASS" \
               ftp.pinproject.com "$@"  )
        echo -E "Trying variant:" 1>&2
        typeset -p -a cmd | sed -r -e 's~^ty[^=]*=\((.*)\)$~\1~; tz; s~^ty[^=]*=~~; 1h; 1!H; $!d; x; s~\n~ ~g; :z s~^ | $~~g; ' 1>&2
        command "${cmd[@]}"
        rs=$?
        echo -E "[ $rs ] with variant=[$variant]" 1>&2
        if (( ! rs )); then
            break
        fi
    done
    [ $rs -eq 0 ] && echo "*** Image upload succeeded: $file ***" 1>&2 
fi

if [ "x$1" == "xupload" ]; then
    echo "Trying some other ideas to publish: [$file]" 1>&2

( cat <<'EOF0'
import codecs, io, os, sys
imgpath = os.getenv("COVERAGE_IMAGE")
with open(imgpath, "r") as f:
  fbytes = f.read()
with open("tmp.json", "w") as jf:
  import json
  jf.write(json.dumps(
    {
      "body": "",
      "description": "",
      "files": {
        "badge.svg": {
          "content":fbytes
        }
      }
    }
  ))

EOF0
) | eval "$PYTHON3"

    pyrs=$?
    curl -X POST -H "Authorization: bearer $GITHUB_TOKEN" \
       -H 'Accept: application/vnd.github.v3+json;q=1.0, */*;q=0.01' \
       "https://api.github.com/gists" -d @tmp.json
    rm -vf -- tmp.json 1>&2
fi


# run coverage!
if [ "x$1" != "xupload" ]; then
    
    yes y | $PYTHON3 -m pip install \
             types-requests || true
    yes y | $PYTHON3 -m pip install --force-reinstall \
             types-requests || true
    yes y | mypy --install-types
    
    covcom=( --cov-branch \
             --cov-report term-missing \
             --cov-report html:coverage/ \
             --cov-report annotate:coverage/annotated \
             --cov-report xml:coverage/cov.xml
    )
    covsrc=( --cov=use --cov=use.use --cov=package_hacks )
    
    mkdir -p ~/.justuse-python
    rm -rf -- ~/.justuse-python/{packages/,registry.json}
    
    if grep -Fqe '=[^="]+=' -- ~/.justuse-python/config.toml; then
        echo -E "Deleting malformed config.toml ..." 1>&2
        rm -vf -- ~/.justuse-python/config.toml
    fi
    
    [ -e ~/.justuse-python/config.toml.bak ] \
      && rm -vf -- ~/.justuse-python/config.toml.bak \
      || true
      
    mv -vf -- ~/.justuse-python/config.toml ~/.justuse-python/config.toml.bak \
      && echo > ~/.justuse-python/config.toml
    
    opts=( -v )
    unset DEBUG ERRORS
    for append in 0 1; do
    
        if (( append > 0 )); then 
          opts+=( --cov-append )
          opts+=( -vv )
          export DEBUG=1 ERRORS=1
        fi
        
        if (( append > 0 )); then 
          echo -E $'\ndebug = true\n\ndebugging = true\n\n'
        else
          echo -E $'\ndebug = false\n\ndebugging = false\n\n'
        fi > ~/.justuse-python/config.toml
          
        $PYTHON3 -m pytest "${covcom[@]}" "${covsrc[@]}" "${opts[@]}"
        rs=$?
        
        (( rs )) && exit $rs
    done
    set +e
    
    ls -lAp --color=always
    find "$cdir" -mindepth 2 -name ".coverage" -printf '%-12s %.10T@ %p\n' | sort -k2n
    find "$cdir" -mindepth 2 -name ".coverage" -printf '%-12s %.10T@ %p\n' \
      | sort -k1n | cut -c 25- | tail -n 1 \
      | {
         max=0
         IFS=$'\n'; while read -r cf; do
             f="$cf"; fn="${f##*/}"
             dir="${f: 0:${#f}-${#fn}}"; dir="${dir%%/}"
             ((max += 1))
             name=".coverage.$max"
             (( max == 1 )) && name="${name%%[!0-9]1}"
             echo -E "Copying cov data from $f to $( pwd )/$name" 1>&2 
             cp -vf -- "$f" "./$name" 1>&2
             cp -vf -- "$f" "./.coverage" 1>&2
         done;
       }
    
    
    mkdir -p coverage
    cp -vf -- .coverage coverage/.coverage
    
    [ -e ~/.justuse-python/config.toml.bak ] \
      && mv ~/.justuse-python/config.toml.bak ~/.justuse-python/config.toml
    
fi

exit ${rs:-0} # upload

exit ${rs:-0} # upload

