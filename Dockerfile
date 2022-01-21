FROM kbase/sdkbase2:python
MAINTAINER KBase Developer
# -----------------------------------------
# In this section, you can install any system dependencies required
# to run your App.  For instance, you could place an apt-get update or
# install line here, a git checkout to download code, or run any other
# installation scripts.

# RUN apt-get update
RUN \
    apt-get update && \
    apt-get -y install gcc && \
    apt-get install -y --reinstall build-essential
RUN pip install --upgrade pip
RUN pip install catboost sklearn pandas matplotlib
RUN pip install git+https://github.com/slundberg/shap.git@b3af833d9c7997994d609df62b1e30834f253469
RUN pip install git+https://github.com/kbase-sfa-2021/sfa@d0ab401a37369e40a84aa62c5276af41253a6139

# -----------------------------------------

COPY ./ /kb/module
RUN mkdir -p /kb/module/work
RUN mkdir -p /opt/work/outputdir
RUN chmod -R a+rw /kb/module

WORKDIR /kb/module

RUN make all

ENTRYPOINT [ "./scripts/entrypoint.sh" ]

CMD [ ]
