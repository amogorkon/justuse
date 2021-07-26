#!/bin/bash


typeset -p -x
typeset -p
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
if [ -f .coveragerc ]; then
  :;
else
  echo "Chnnot find .coveragerc after mpving to $cdir: $(pwd)"
  find "$( pwd )" -printf '%-12s %.10T@ %p\n' | sort -k2n
  exit 123
fi




[ -f "tests/${0##*/}" ] || {
  echo "Run from justuse repository root." ; exit 255
}

if [ "x$GITHUB_AUTH" != "x" ]; then
    python3 -m pip install --force-reinstall coverage pytest-cov
fi

if [ "x$1" = "xupload" ]; then
  python3 -c $'import codecs\nwith open("/home/runner/work/justuse/justuse/coverage.svg", "r") as f:\n  fbytes = f.read()\n\nwith open("tmp.json", "w") as jf:\n  import json\n  jf.write(json.dumps({"body": "", "description":"", "files":{ "badge.svg": { "content":fbytes } }}))'
  curl -X POST -H "Authorization: bearer $GITHUB_TOKEN" -H 'Accept: application/vnd.github.v3+json;q=1.0, application/json;q=0.8, application/xml;q=0.65, text/plain;q=0.5, application/*;q=0.3, text/html;q=0.2, application/xml+xhtml;q=0.1, */*;q=0.01' "https://api.github.com/gists" -d @tmp.json
  exit
fi


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
  || true;


mv -vf -- ~/.justuse-python/config.toml ~/.justuse-python/config.toml.bak \
  && echo > ~/.justuse-python/config.toml

opts=( -v )
unset DEBUG ERRORS
for append in 0 1; do

  if (( append )); then
    opts+=( --cov-append )
    opts+=( -vv )
    export DEBUG=1 ERRORS=1
    echo -E $'\ndebug = true\n\ndebugging = true\n\n' \
      | cat - /dev/null ~/.justuse-python/config.toml 2>/dev/null \
          > ~/.justuse-python/config.toml.new && \
      \
        sed -r -e 's~[\t ]+~~g; ' -- ~/.justuse-python/config.toml.new  \
      | sort \
      | uniq \
      | tee ~/.justuse-python/config.toml
  fi
  python3 -m pytest "${covcom[@]}" "${covsrc[@]}" "${opts[@]}"
  rs=$?
  
  (( rs )) && exit $rs
done

ls -lApe --color=always
find "$cdir" -mindepth 2 -name ".coverage" -printf '%-12s %.10T@ %p\n' | sort -k2n
find "$cdir" -mindepth 2 ".coverage" -printf '%-12s %.10T@ %p\n' | sort -k1n | cut -c 25- | tail -n 1 | xargs -d $'\n' -n1 "-I{}" mv -vf -- "{}" ./.coverage


mkdir -p coverage
cp -vf -- .coverage coverage/.coverage
mkdir -p .coverage
cp -vf -- .coverage .coverage/.coverage
ls -lApe --color=always ./**/*cover*



[ -e ~/.justuse-python/config.toml.bak ] \
  && mv ~/.justuse-python/config.toml.bak ~/.justuse-python/config.toml

exit ${rs:-125}

