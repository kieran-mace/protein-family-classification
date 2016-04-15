dat = read.table('data/silver_standard_all_matrix.txt')
dat = dat[,-1]
vec = as.vector(as.matrix(dat))
pdf('scores.pdf')
hist(vec,breaks=1000,xlim=c(-10,600),main='all data')
hist(vec[vec<50],breaks=1000,xlim=c(-10,50),main='all data smaller than 50')
dev.off()
