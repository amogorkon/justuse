#!/bin/bash
set -o allexport
arg1="$1"

[ -d use -a -f use/use.py ] && cd ..
[ -f unit_teat.py ] && cd ..
while [ ! -e requirements.txt ]; do
  cd ..
done

eval "cov_file_opts=($( find -name "*.py" | sed -r -e 's~^\./~~; s~\.py$~~; s~[_-][^_/.-]*\..*$~~; s~/~.~g; s~^src\.~~; s~^~--cov=~;' | sort | tr -s '\n ' " "; ))"

: ${PYTHON:=python3}
[ "x${PYTHON: 0:1}" = "x/" ] || PYTHON="$( which ""$PYTHON"" || which "python" || which "python.exe" )"
export PYTHON

if [ "x$GITHUB_AUTH$GITHUB_PATH$GITHUB_ROOT$GITHUB_USER$GITHUB_AUTHOR$GITHUB_REPO$GITHUB_COMMIT$GITHUB_REF$GITHUB_UID$GITHUB$USERID" != "x" ]; then
  find . -regextype egrep "(" "(" -name .git -prune -false ")" -o -iregex '.*/\..*cache.*|.*cache.*/\..*' ")" -a -exec rm -vrf -- "{}" +
  yes y | "$PYTHON" -m pip install --force-reinstall --upgrade -r requirements.txt
fi
which apt && ! which busybox && { apt install -y busybox || sudo apt install -y busybox; }


echo "=== COVERAGE SCRIPT ==="
echo "$0 running from $( pwd )"
coveragerc_cdir="$( pwd )"
coveragerc_cfile="$coveragerc_cdir/.coveragerc"
while [ ! -e "$coveragerc_cfile" ]; do
  coveragerc_cdir="${coveragerc_cfile%/?*}"
  coveragerc_cdir2="${coveragerc_cdir%/?*}"  
  [ "x${coveragerc_cfile2##/}" = "x/" ] && echo "No coveragerc" && exit 255
  coveragerc_cfile="$( find "$coveragerc_cdir2" -mindepth 2 -maxdepth 4 -name .coveragerc )"
done
coveragerc_cdir="${coveragerc_cfile%/?*}"
  
echo "coveragerc_cfile=${coveragerc_cfile}" 1>&2
echo "coveragerc_cdir=${coveragerc_cdir}" 1>&2
! cd "$coveragerc_cdir" && echo "Failed to cd to \$coveragerc_dir [\"$coveragerc_dir\"]" 1>&2 && exit 255



IFS=$'\n'; badge_urls=($( curl -ks "https://raw.githubusercontent.com/amogorkon/justuse/main/README.md" | grep -e '\[!\[coverage\]' | tr -s '()' '\n' | grep -Fe "://" | grep -Fe ".svg" )); badge_url_no_query="${badge_urls[0]%%[#\?]*}"; badge_fn="${badge_url_no_query##*/}"; BADGE_FILENAME="$badge_fn";
default="/home/runner/work/justuse/justuse/$BADGE_FILENAME"
f="$default"; fn="${f##*/}"; dir="${f: 0:${#f}-${#fn}}"; dir="${dir%%/}"
[ ! -d "$dir" ] && default="$PWD/coverage/$BADGE_FILENAME"
file="${COVERAGE_IMAGE:-${default}}"
f="$file"; fn="${f##*/}"; dir="${f: 0:${#f}-${#fn}}"; dir="${dir%%/}"; nme="${fn%.*}"
covcom=( --cov-branch --cov-report term-missing --cov-report html:coverage/ --cov-report annotate:coverage/annotated --cov-report xml:coverage/cov.xml )
covsrc=( "${cov_file_opts[@]}" )
  
echo "default=$default" 1>&2
echo "COVERAGE_IMAGE=$COVERAGE_IMAGE" 1>&2
echo "file=$file" 1>&2
  
opts=( -v )
for append in 1; do

  if (( append > 0 )); then 
    opts+=( --cov-append )
    opts+=( -vv )
  fi
  
    
  "$PYTHON" -m pytest "${covcom[@]}" "${covsrc[@]}" "${opts[@]}"
  rs=$?
  
  (( rs )) && exit $rs
done
set +e

if [ ! -f .coveragerc ]; then
  echo "Cannot find .coveragerc after mpving to $coveragerc_cdir: $(pwd)"
  find "$( pwd )" -printf '%-12s %.10T@ %p\n' | sort -k2n
  exit 123
fi

ls -lAp --color=always
find "$coveragerc_cdir" -mindepth 2 -name ".coverage" -printf '%-12s %.10T@ %p\n' \
  | sort -k1n | cut -c 25- | tail -n 1 \
  | {
    max=0
    IFS=$'\n'; while read -r coveragerc_cf; do
      f="$coveragerc_cf"; fn="${f##*/}"
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

mkdir -p "$dir"
python3 -m coverage_badge | tee "$file"
echo "Found an image to publish: [$file]" 1>&2
orig_file="$file"

IFS=$'\n'; remotes=($( git remote -v  | cut -f2 | cut -d: -f2 | cut -d/ -f1 | sort | uniq ))

branch="$( git branch -v | grep -Fe "*" | head -1 | cut -d " " -f2; )"
badge_filenames=( )
for remote in "${remotes[@]}"; do
  badge_filename="coverage_${remote}-${branch}.svg"
  badge_filenames+=( "$badge_filename" )
done
 
python3 -m pip install coverage-badge
if [ "x$FTP_USER" != "x" -a $( python3 -m coverage_badge | wc -c ) -gt 800 ]; then
  python3 -c "import coverage_badge" >/dev/null 2>&1 || python3 -m pip install --force-reinstall coverage-badge
  for filename in "$orig_file" "${badge_filenames[@]}"; do
    f="$file"; fn="${f##*/}"; dir="${f: 0:${#f}-${#fn}}"; dir="${dir%%/}"; _dir="$dir"; f="$filename"; fn="${f##*/}"; dir="${f: 0:${#f}-${#fn}}"; dir="${dir%%/}"; _fn="$fn"; f="$file"; fn="${f##*/}"
    fn="${filename##*/}"
    rm -vf -- "$file" || rmdir "$file"; python3 -m coverage_badge | cat -v | tee "$_dir/$_fn" | tee "$file"; 
    for variant in \
      '"/public_html/mixed/$fn" "$file"'  \
      '"/public_html/mixed/coverage_amogorkon-main.svg" "$file"' \
      '"/public_html/mixed/coverage.svg" "$file"' \
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

