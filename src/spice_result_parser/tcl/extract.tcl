# extract.tcl: write every point of selected signals of one waveform file
# (WaveView batch mode).
#
# Part of spice_result_parser (waveview.py runs it). Run a copy of it by hand in
# a working directory (WaveView writes ace_tcl.log beside the script):
#   Linux:   SRP_IN=run.tr0 SRP_OUT=out.txt wv -ace_no_gui extract.tcl
#   Windows: set SRP_IN=run.tr0&& set SRP_OUT=out.txt&& wv.exe -ace_no_gui extract.tcl
#
# The points come from a POINT scan, not from sx_export_data: the export drops
# points that repeat an x value (frequent in float32 .tr0 files) and prints only
# 16 digits. The scan gives every point of the file with 17 digits.
#
# Input:  SRP_IN      the waveform file.
#         SRP_SIGNALS optional: a text file with one exact signal name per line,
#                     as list_traces.tcl writes them. Not set: every signal.
#                     No names in it: x only.
#         SRP_XCHECK  optional: set, the script checks that all signals have
#                     the same x points (see write_points).
# Output: SRP_OUT     a text file. Records have a tab; value lines do not:
#   srp_extract 1
#   trace   <i>           0-based; one per sweep trace, one when not swept
#   x       <signal>      then the x value of each point, one per line
#   real    <signal>      then the y value of each point, one per line
#   complex <signal>      then "<real part> <imaginary part>" per point
#   ...                   (the signals in the order of the file)
#   srp_done
# NaN prints as 1.#QNAN on Windows. A file that does not open, a signal name
# that the file does not have, or signals with their own x points (an HSPICE
# post=3 file) give "error<TAB><text>" then srp_done. A missing srp_done means
# that the script failed: the console (stdout) has the WaveView error. The exit
# code does not show success: it is 0 after an error that the script catches.
#
# Tcl 8.4: no dict, lassign, {*} or "in".

set tcl_precision 17
set out [open $env(SRP_OUT) w]
fconfigure $out -encoding utf-8 -buffersize 1048576

proc finish {{error ""}} {
    global out
    if {$error != ""} {
        puts $out "error\t$error"
    }
    puts $out srp_done
    close $out
    exit
}

# A name as valid UTF-8. WaveView gives the bytes of the file: a Latin-1 byte
# would not equal the name in SRP_SIGNALS, and puts can lose the end of the line.
proc clean {name} {
    return [encoding convertfrom utf-8 [encoding convertto utf-8 $name]]
}

# Write one value line per point of trace `trace` of signal `sig`. With
# SRP_XCHECK, the x scan keeps its values in xs, and a y scan compares its x
# values with them: all the signals of the trace must have the same x points.
# The check costs time, so waveview.py sets SRP_XCHECK only for a file whose
# format does not give one x to all signals. The loops are in a proc so that
# Tcl compiles them.
proc write_points {sig trace swept kind} {
    global out xs xname xcheck
    set scan [sx_scan_data_points $sig POINT]
    if {$swept} {
        set more [sx_jump_to_x $scan FIRST -trace $trace]
    } else {
        set more [sx_jump_to_x $scan FIRST]
    }
    set k 0
    set n [llength $xs]
    if {$kind == "x" && $xcheck} {
        set xs {}
        while {$more == 1} {
            set x [sx_x_value $scan]
            lappend xs $x
            puts $out $x
            set more [sx_jump_to_x $scan NEXT]
        }
    } elseif {$kind == "x"} {
        while {$more == 1} {
            puts $out [sx_x_value $scan]
            set more [sx_jump_to_x $scan NEXT]
        }
    } elseif {$kind == "complex"} {
        while {$more == 1} {
            if {$k < $n && [sx_x_value $scan] != [lindex $xs $k]} {
                break
            }
            puts $out "[sx_y_value $scan real] [sx_y_value $scan imag]"
            incr k
            set more [sx_jump_to_x $scan NEXT]
        }
    } else {
        while {$more == 1} {
            if {$k < $n && [sx_x_value $scan] != [lindex $xs $k]} {
                break
            }
            puts $out [sx_y_value $scan]
            incr k
            set more [sx_jump_to_x $scan NEXT]
        }
    }
    sx_release $scan
    if {$more == 1} {
        finish "the signals have their own x points (an HSPICE post=3 file, for example):\
            [clean [sx_get_name $sig]] has other x points than $xname; extract one signal at a time"
    }
}

# sx_open_sim_file_read splits its argument as a Tcl list: pass one element, in
# quotes when the path has a brace ([list] escapes a lone brace with a backslash,
# which WaveView changes to "/"). It cannot open a path with non-ASCII characters
# or a Windows path of 260 characters or more: try the 8.3 short name.
set path $env(SRP_IN)
if {![string is ascii $path] || [string length $path] >= 260} {
    catch {set path [file attributes $path -shortname]}
}
set arg [list $path]
if {[regexp {[{}]} $path] && [string first "\"" $path] < 0} {
    set arg "\"$path\""
}
set files {}
catch {set files [sx_open_sim_file_read $arg]}
set f [lindex $files 0]
set why ""
if {![string is ascii $path]} {
    set why ": WaveView cannot open a path with non-ASCII characters"
} elseif {[string length $path] >= 260} {
    set why ": WaveView cannot open a path of 260 characters or more"
}
if {$f == ""} {
    finish "cannot open $env(SRP_IN)$why"
}
sx_current_sim_file $f
set all [sx_signal *]
set order {}
foreach s $all {
    set name [clean [sx_get_name $s]]
    set object($name) $s
    set nameof($s) $name
    lappend order $name
}
if {[llength $all] == 0} {
    finish "the file has no signals"
}

if {[info exists env(SRP_SIGNALS)]} {
    set in [open $env(SRP_SIGNALS) r]
    fconfigure $in -encoding utf-8
    while {[gets $in line] >= 0} {
        set name [string trimright $line "\r"]
        if {$name == ""} {
            continue
        }
        if {![info exists object($name)]} {
            close $in
            finish "unknown signal $name"
        }
        set wanted($name) 1
    }
    close $in
} else {
    foreach name $order {
        set wanted($name) 1
    }
}
set sigs {}
foreach name $order {
    if {[info exists wanted($name)]} {
        lappend sigs $object($name)
    }
}

set ntrace [llength [sx_get_sweep_conditions $f]]
set swept [expr {$ntrace > 0}]
if {!$swept} {
    set ntrace 1
}
set xsig [lindex $all 0]
if {[llength $sigs] > 0} {
    set xsig [lindex $sigs 0]
}
set xname $nameof($xsig)
set xs {}
set xcheck [info exists env(SRP_XCHECK)]
puts $out "srp_extract\t1"
for {set i 0} {$i < $ntrace} {incr i} {
    puts $out "trace\t$i"
    puts $out "x\t$xname"
    write_points $xsig $i $swept x
    foreach s $sigs {
        set kind [sx_signal_attribute $s -datatype]
        if {$kind != "complex"} {
            set kind real
        }
        puts $out "$kind\t$nameof($s)"
        write_points $s $i $swept $kind
    }
}
finish
