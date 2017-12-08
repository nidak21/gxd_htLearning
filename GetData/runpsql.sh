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
    -c|--cleanse)  clean="y"; shift; ;;
    --align) alignopt=""; shift; ;;
    -v|--verbose) quietopt=""; shift; ;;
    --) shift; break; ;;
    -*|--*) echo "invalid option $1"; Usage ;;
    *) break; ;;
    esac
done

cat -  >| $sqltmp
if [ "$quietopt" == "" ]; then
    echo "hitting $server, database $db as mgd_public"  >&2
fi
psql -h $server -d $db -U mgd_public -f $sqltmp -F $'\11' $alignopt $quietopt "$@" >| $outputtmp

if [ "$quietopt" == "-q" ]; then
    # note: if --align, this removes the blank line at end, not the count line
    head -n -1 $outputtmp	# remove last line
else
    cat $outputtmp
fi
rm $outputtmp $sqltmp
