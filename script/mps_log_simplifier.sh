#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-only


if [[ -f $1 ]]; then
  cat $1 | egrep -i '(Site|sweep|Simu)'
else
  egrep -i '(Site|sweep|Simu)'
fi
