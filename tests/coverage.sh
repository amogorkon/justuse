#!/bin/bash
arg1="$1"

[ -d use -a -f use/use.py ] && cd ..
[ -f unit_teat.py ] && cd ..
while [ ! -e requirements.txt ]; do
  cd ..
done

: ${PYTHON3:=python3}
export PYTHON3


if [ "x$GITHUB_AUTH$GITHUB_PATH$GITHUB_ROOT$GITHUB_USER$GITHUB_AUTHOR$GITHUB_REPO$GITHUB_COMMIT$GITHUB_REF$GITHUB_UID$GITHUB$USERID" != "x" ]; then
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
set +e
BADGE_FILENAME="coverage.svg"
default="/home/runner/work/justuse/justuse/$BADGE_FILENAME"
f="$default"; fn="${f##*/}"; dir="${f: 0:${#f}-${#fn}}"; dir="${dir%%/}"
if [ -d "$dir" ]; then
  :
else
  default="$PWD/coverage/$BADGE_FILENAME"
fi
echo "default=$default" 1>&2
file="$default"
echo "COVERAGE_IMAGE=$COVERAGE_IMAGE" 1>&2
file="${COVERAGE_IMAGE:-${default}}"
echo "file=$file" 1>&2

export COVERAGE_IMAGE
f="$file"
fn="${f##*/}"
dir="${f: 0:${#f}-${#fn}}"; dir="${dir%%/}"
nme="${fn%.*}"
if [ ! -f "$file" ]; then

# run coverage!
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
for append in 1; do

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

if [ ! -f .coveragerc ]; then
  echo "Cannot find .coveragerc after mpving to $cdir: $(pwd)"
  find "$( pwd )" -printf '%-12s %.10T@ %p\n' | sort -k2n
  exit 123
fi

ls -lAp --color=always
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


f="$file"; fn="${f##*/}"; dir="${f: 0:${#f}-${#fn}}"; dir="${dir%%/}"
mkdir -p "$dir"
python3 -m coverage_badge | tee "$file"
echo "Found an image to publish: [$file]" 1>&2


orig_file="$file"
BADGE_FILENAME="$( ( git remote -v | cut -f2 | sed -r -e 's~[\t ].*$~~; s~^.*\.(net|edu|com|org)[:/]~~; s~\.git$~~; s~/~-~g; ' | head -1; git branch -v -a | grep -Fe "*" | cut -d " " -f2; ) | tr -s $'\n ' '.' | sed -r -e 's~\.*$~~; s~^~coverage_~; '; echo -n ".svg"; )";
 
if [ $( python3 -m coverage_badge | wc -c ) -gt 800 ]; then
  python3 -c "import coverage_badge" >/dev/null 2>&1 || python3 -m pip install --force-reinstall coverage-badge
  for filename in "$orig_file" "$BADGE_FILENAME"; do
      f="$file"; fn="${f##*/}"; dir="${f: 0:${#f}-${#fn}}"; dir="${dir%%/}"; _dir="$dir"; f="$filename"; fn="${f##*/}"; dir="${f: 0:${#f}-${#fn}}"; dir="${dir%%/}"; _fn="$fn"; f="$file"; fn="${f##*/}"
      fn="${filename##*/}"
      rm -vf -- "$file" || rmdir "$file"; python3 -m coverage_badge | cat -v | tee "$_dir/$_fn" | tee "$file"; 
      for variant in \
          '"/public_html/mixed/$fn" "$file"'  \
          ;  \
      do
          eval "set -- $variant"
          cmd=(  busybox ftpput -v -P 21 -u "$FTP_USER" -p "$FTP_PASS" \
                ftp.pinproject.com "$@"  )
          echo -E "Trying variant:" 1>&2
          if (( ! UID )); then
            echo "$@" 1>&2
          fi
          command "${cmd[@]}"
          rs=$?
          if (( ! rs )); then
              break
          fi
      done
  [ $rs -eq 0 ] && echo "*** Image upload succeeded: $@ ***" 1>&2 
  done
fi

exit ${rs:-0} # upload

