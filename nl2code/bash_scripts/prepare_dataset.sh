#! /bin/bash -x

python -c "from nl2code.dataset import parse_django_dataset; parse_django_dataset()"
