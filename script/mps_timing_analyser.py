#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-06-14 13:29
#  
#  Description: GraceQ/mps2 project. Timing log analysis tool.
# 
import sys
import re
from collections import namedtuple


SPLIT_WEIGHT = 96


class RoutTimColl(object):

  """Routine timing collection. """

  def __init__(self, routine, times, tot, max, min):
    self.routine = routine
    self.times = times
    self.tot = tot
    self.max = max
    self.min = min

  def get_avg_time(self):
    self.avg = self.tot / self.times


def print_sample_analy_header(log_file):
  """Print sample analysis header. """
  print('\n\n\n' + '*' * SPLIT_WEIGHT +'\n'+ '*' * SPLIT_WEIGHT)
  print(log_file.rstrip())
  print('\n'.rstrip())
  print('{0:25} {1:5} {2:>15} {3:>15} {4:>15} {5:>15}'.format(
        'routine', 'times', 'tot time', 'avg time', 'max time', 'min time'))
  print('*' * SPLIT_WEIGHT)
  

def clean_log_data(file_lines):
  """Get timing log from original log file. """
  pattern = r'^\[timing\]\s+(\S+)\s+(\S+)\s+$'
  regex = re.compile(pattern)
  mo_list = list(regex.search(line) for line in file_lines)
  mo_list = list(mo for mo in mo_list if mo is not None)
  data_filtered = list(
        list(str_.strip() for str_ in mo.groups()) for mo in mo_list)

  return list([timing_log_item[0], float(timing_log_item[1])] for
            timing_log_item in data_filtered)


def timing_profile_analyser(timing_log):
  """Analysis timing log. """
  routines = []
  routtimcolls = {}
  for item in timing_log:
    routine = item[0]
    elag_time = item[1]
    if routine in routines:
      routtimcoll = routtimcolls[routine]
      routtimcoll.times += 1
      routtimcoll.tot += elag_time
      routtimcoll.max = max(routtimcoll.max, elag_time)
      routtimcoll.min = min(routtimcoll.min, elag_time)
    else:
      routines.append(routine)
      routtimcolls.update({routine: RoutTimColl(routine, 1,
                                                elag_time,
                                                elag_time,
                                                elag_time)})
  for _, routtimcoll in routtimcolls.items():
    routtimcoll.get_avg_time()
    print('{0:25} {1:5d} {2:15.3f} {3:15.3f} {4:15.3f} {5:15.3f}'.format(
                                                           routtimcoll.routine,
                                                           routtimcoll.times,
                                                           routtimcoll.tot,
                                                           routtimcoll.avg,
                                                           routtimcoll.max,
                                                           routtimcoll.min))


def print_sample_analy_tailer():
  """Print sample analysis tailer. """
  print('*' * SPLIT_WEIGHT)


if __name__ == '__main__':
  from_stdin = False
  if len(sys.argv) > 1:
    log_file_path = sys.argv[1]
    log_file_name = log_file_path.split('/')[-1]
    log_file = open(log_file_path, 'r')
  else:
    log_file = sys.stdin
    log_file_name = 'stdin'
    from_stdin = True
  timing_log = clean_log_data(log_file.readlines())
  if not from_stdin:
    log_file.close()

  SPLIT_WEIGHT = max(SPLIT_WEIGHT, len(log_file_name)+1)
  print_sample_analy_header(log_file_name)

  timing_profile_analyser(timing_log)

  print_sample_analy_tailer()
