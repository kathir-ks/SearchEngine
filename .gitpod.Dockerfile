FROM gitpod/workspace-full

USER gitpod

RUN brew tap mongodb/brew && brew install mongodb-community@4.4

USER root