# Lightning Sample project/package

This is starter project template which shall simplify initial steps for each new PL project...

[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![CI testing](https://github.com/Lightning-AI/lightning-sandbox/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-sandbox/actions/workflows/ci-testing.yml)
[![General checks](https://github.com/Lightning-AI/lightning-sandbox/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-sandbox/actions/workflows/ci-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/lightning-sandbox/badge/?version=latest)](https://lightning-sandbox.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-sandbox/main.svg?badge_token=mqheL1-cTn-280Vx4cJUdg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-sandbox/main?badge_token=mqheL1-cTn-280Vx4cJUdg)

\* the Read-The-Docs is failing as this one leads to the public domain which requires the repo to be public too

## Included

Listing the implemented sections:

- sample package named `pl_sandbox`
- setting [CI](https://github.com/Lightning-AI/lightning-sandbox/actions?query=workflow%3A%22CI+testing%22) for package and _tests_ folder
- setup/install package
- setting docs with Sphinx
- automatic PyPI release on GH release
- Docs deployed as [GH pages](https://Lightning-AI.github.io/lightning-sandbox)
- Makefile for building docs with `make docs` and run all tests `make test`

## To be Done aka cross-check

You still need to enable some external integrations such as:

- [ ] rename `pl_<sandbox>` to anu other name, simple find-replace shall work well
- [ ] update path used in the badges to the repository
- [ ] lock the main breach in GH setting - no direct push without PR
- [ ] set `gh-pages` as website and _docs_ as source folder in GH setting
- [ ] init Read-The-Docs (add this new project)
- [ ] add credentials for releasing package to PyPI
- [ ] specify license in `LICENSE` file and package init

## Tests / Docs notes

- We are using [Napoleon style,](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) and we shall use static types...
- It is nice to se [doctest](https://docs.python.org/3/library/doctest.html) as they are also generated as examples in documentation
- For wider and edge cases testing use [pytest parametrization](https://docs.pytest.org/en/stable/parametrize.html) :\]
