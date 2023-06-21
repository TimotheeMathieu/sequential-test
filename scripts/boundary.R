#!/usr/bin/env Rscript

list.of.packages <- c("ldbounds", "optparse")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library("ldbounds")
library("optparse")

 
option_list = list(
  make_option(c("-n", "--name"), type="character", default="PK",
              help="name of bound. OF for O'Brien-Fleming, PK for Pocock.
name values of 1 and 2 correspond to alpha spending functions which give O’Brien Fleming and
Pocock type boundaries, respectively. A value of 3 is the power family. Here, the spending function
is αtφ, where φ must be greater than 0. A value of 4 is the Hwang-Shih-DeCani family, with
spending function α(1 − e−φt)/(1 − e−φ), where φ cannot be 0."),
  make_option(c("-a", "--alpha"), type="double", default=0.05, 
              help="Level of the test, type I error."),
  make_option(c("-l", "--looks"), type="integer", default=5 , 
              help="A number of equally spaced analysis times"),
  make_option(c("-s", "--sides"), type="integer", default=2, 
              help="two sided (2) or one-sided (1) test")
)
 
opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)
t = seq(0,1, length.out=opt$looks+1)[seq(2,opt$looks+1)]

if ((opt$name == "OF") || (opt$name =="PK")){
bd <- commonbounds(t=t, iuse=opt$name, alpha=opt$alpha, sides=opt$sides)
}else{
bd <- ldBounds(t=t, iuse=opt$name, alpha=opt$alpha, sides=opt$sides)
}

df1 = read.table('boundaries.csv', sep=";")

df2 = data.frame(up=bd$upper.bounds, alpha=opt$alpha, K=opt$looks, name=opt$name, t=t)
df = rbind(df1, df2)
df = df[!duplicated(df), ]
write.table(df, 'boundaries.csv', sep=";")
