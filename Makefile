PG_CONFIG ?= pg_config
PKGLIBDIR := $(shell $(PG_CONFIG) --pkglibdir)
SHAREDIR := $(shell $(PG_CONFIG) --sharedir)
MKDIR ?= mkdir
CP ?= cp

.PHONY: all clippy build install uninstall

all: build

clippy:
	PGRX_PG_CONFIG_PATH="$(PG_CONFIG)" cargo run -p xtask --bin xtask_clippy

build:
	PGRX_PG_CONFIG_PATH="$(PG_CONFIG)" cargo run -p xtask --bin xtask_build

install:
	$(MKDIR) -p $(DESTDIR)$(PKGLIBDIR) $(DESTDIR)$(SHAREDIR) && \
	$(CP) -r ./build/pkglibdir/. $(DESTDIR)$(PKGLIBDIR) && \
	$(CP) -r ./build/sharedir/. $(DESTDIR)$(SHAREDIR)

uninstall:
	$(RM) $(wildcard $(DESTDIR)$(PKGLIBDIR)/vchord.*) && \
	$(RM) $(wildcard $(DESTDIR)$(SHAREDIR)/extension/vchord.*) && \
	$(RM) $(wildcard $(DESTDIR)$(SHAREDIR)/extension/vchord--*.sql)
