#!/bin/bash
# get training data for gxd_htLearning machine learning project
function Usage() {
    s=$(basename $0)
    cat - <<ENDTEXT

$s [-s server] [-d database] [options...] [--] [options to psql]

    Reads sql query from stdin, runs it with psql, writes results to stdout.
    Extra psql output lines are removed. Options:

    -s, --server  SERVER	db server
				shorthands: adhoc (default), prod, dev
    -d, --database DATABASE	database. Default: "mgd"
    -c, --cleanse		remove non-ascii chars from the output (default)
    --dirty			leave non-ascii chars
    --align			psql aligned format. Default: --no-align
    -v, --verbose		unquiet psql. Default: -q, quiet
    --				pass the following arguments to psql cmd line
  psql opts: --no-align, -F fieldsep, -R recordsep, -t (tuples only), -q (quiet)
ENDTEXT
    exit 5
}
# Defaults
sqltmp=tmpsqlfile
outputtmp=tmpoutput
server="mgi-adhoc"
db="mgd"
alignopt="--no-align"
quietopt="-q"
clean="true"
cleanseCmd="env LANG=C tr -cd '[:print:]\n\t'"
headCmd="head -n -1"

while [ $# -gt 0 ]; do
    case "$1" in
    -h|--help) Usage ;;
    -s|--server) 
	    case "$2" in
	    adhoc) server="mgi_adhoc"; db="mgd" ;;
	    prod) server="bhmgidb01"; db="prod" ;;
	    dev) server="bhmgidevdb01"; db="prod" ;;
	    *) server="$2" ;;
	    esac
            shift; shift;
            ;;
    -d|--database) db="$2"; shift; shift; ;;
    -c|--cleanse)  clean="true"; shift; ;;
    --dirty) clean="false"; shift; ;;
    --align) alignopt=""; shift; ;;
    -v|--verbose) quietopt=""; shift; ;;
    --) shift; break; ;;
    -*|--*) echo "invalid option $1"; Usage ;;
    *) break; ;;
    esac
done

cat -  >| $sqltmp

echo "hitting $server, database $db as mgd_public"  >&2

cmd="psql -h $server -d $db -U mgd_public -f $sqltmp -F \$'\11' $alignopt $quietopt "$@"" 
if [ "$clean" == "true" ]; then
    cmd="$cmd | $cleanseCmd"
fi
if [ "$quietopt" == "-q" ]; then
    cmd="$cmd | $headCmd"
fi
#set -x
eval $cmd
#set +x

rm $sqltmp
