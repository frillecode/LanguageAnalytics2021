#!/usr/bin/env bash

VENVNAME=cds-lang
jupyter kernelspec uninstall $VENVNAME
rm -r $VENVNAME