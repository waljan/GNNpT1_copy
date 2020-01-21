#!/usr/bin/R

get_test_data = function(folder, m)
{
    path = paste("out/", folder, m, "/", m, "_test_data.csv", sep="")
    df = read.csv(path, header=F, sep=",", fill=TRUE)
    return(df)
}


plot_Results = function()
{
    # pdf(file="Results_all_GNNs.pdf", height=10/2.54, width=20/2.54)

    folders = list("base/", "paper/")
    models = list("GraphSAGE", "GCN", "GraphSAGEWithJK", "GATNet", "GCNWithJK","OwnGraphNN2")#,"OwnGraphNN", "NMP")
    num_models = length(models)
    total_sds = vector(length=2*num_models)
    total_avgs = vector(length=2*num_models)
    fold0_sds = vector(length=2*num_models)
    fold1_sds = vector(length=2*num_models)
    fold2_sds = vector(length=2*num_models)
    fold3_sds = vector(length=2*num_models)
    fold0_avgs = vector(length=2*num_models)
    fold1_avgs = vector(length=2*num_models)
    fold2_avgs = vector(length=2*num_models)
    fold3_avgs = vector(length=2*num_models)

    c = 0
    for(folder in folders)
    {
        for(m in models)
        {
            c=c+1
            df = get_test_data(folder,m)
            total_sds[c]=df[1,1]
            total_avgs[c]=df[1,2]
            fold0_sds[c]=df[2,1]
            fold0_avgs[c]=df[2,2]
            fold1_sds[c]=df[3,1]
            fold1_avgs[c]=df[3,2]
            fold2_sds[c]=df[4,1]
            fold2_avgs[c]=df[4,2]
            fold3_sds[c]=df[5,1]
            fold3_avgs[c]=df[5,2]
        }
    }
    df = get_test_data("", "CNN")
    cnn_sds = c(df[1,1],df[2,1],df[3,1],df[4,1],df[5,1])
    cnn_avgs = c(df[1,2],df[2,2],df[3,2],df[4,2],df[5,2])


    x = 1:num_models
    x2 = 1:(num_models+1)
    x_max= num_models+2
    offset = 0.1
    colors_base="medium violet red"
    colors_base_0 = "red"
    colors_base_1 = "dark orange"
    colors_base_2 = "gold"
    colors_base_3 = "yellow"

    colors_paper="medium blue"
    colors_paper_0 = "dodger blue"
    colors_paper_1 = "cyan"
    colors_paper_2 = "medium turquoise"
    colors_paper_3 = "dark cyan"



    # draw plot
    plot(NULL, xlim=c(0.7, x_max-0.3), ylim=c(0.65, 1), xaxt="n", xlab="", ylab="test accuracy", cex.axis=0.5, cex.lab=0.5)
    rect(par("usr")[1],par("usr")[3],par("usr")[2],par("usr")[4],col = "white smoke")
    rect(xleft=x2-0.05, xright=x2+0.05, ybottom=par("usr")[3], ytop=par("usr")[4], col="white", border="grey", lwd=0.5)
    rect(xleft=x2+2*offset-0.05, xright=x2+2*offset+0.05, ybottom=par("usr")[3], ytop=par("usr")[4], col="white", border="grey", lwd=0.5)
    rect(xleft=x2+4*offset-0.05, xright=x2+4*offset+0.05, ybottom=par("usr")[3], ytop=par("usr")[4], col="white", border="grey",lwd=0.5)
    abline(h=seq(0.65,1,0.01), lty="dotted", col="grey",lwd=0.5)
    abline(h=seq(0.65,1,0.05), lty="dashed", col="grey",lwd=0.5)
    abline(v=seq(1.7, num_models-0.3,1), col="grey",lwd=0.5)
    # draw total sd of every model
    arrows(x0=x-0.01, x1=x-0.01, y0=head(total_avgs,num_models)-head(total_sds,num_models), y1=head(total_avgs,num_models)+head(total_sds,num_models), code=3, angle=90, len=0.02, col=colors_base, lwd=1)
    arrows(x0=x+0.01, x1=x+0.01, y0=tail(total_avgs,num_models)-tail(total_sds,num_models), y1=tail(total_avgs,num_models)+tail(total_sds,num_models), code=3, angle=90, len=0.02, col=colors_paper, lwd=1)

     # draw sd of CNN
    for(i in 1:5)
    {
        arrows(x0=tail(x2,1)+(i-1)*offset, x1=tail(x2,1)+(i-1)*offset, y0=cnn_avgs[i]-cnn_sds[i],y1=cnn_avgs[i]+cnn_sds[i], code=3, angle=90, len=0.02, lwd=1)
    }

    # draw sd for every fold and every model
    arrows(x0=x+offset-0.01, x1=x+offset-0.01, y0=head(fold0_avgs,num_models)-head(fold0_sds,num_models), y1=head(fold0_avgs,num_models)+head(fold0_sds,num_models), code=3, angle=90, len=0.02, col=colors_base_0, lwd=1)
    arrows(x0=x+offset+0.01, x1=x+offset+0.01, y0=tail(fold0_avgs,num_models)-tail(fold0_sds,num_models), y1=tail(fold0_avgs,num_models)+tail(fold0_sds,num_models), code=3, angle=90, len=0.02, col= colors_paper_0, lwd=1)

    arrows(x0=x+2*offset-0.01, x1=x+2*offset-0.01, y0=head(fold1_avgs,num_models)-head(fold1_sds,num_models), y1=head(fold1_avgs,num_models)+head(fold1_sds,num_models), code=3, angle=90, len=0.02, col=colors_base_1, lwd=1)
    arrows(x0=x+2*offset+0.01, x1=x+2*offset+0.01, y0=tail(fold1_avgs,num_models)-tail(fold1_sds,num_models), y1=tail(fold1_avgs,num_models)+tail(fold1_sds,num_models), code=3, angle=90, len=0.02, col= colors_paper_1, lwd=1)

    arrows(x0=x+3*offset-0.01, x1=x+3*offset-0.01, y0=head(fold2_avgs,num_models)-head(fold2_sds,num_models), y1=head(fold2_avgs,num_models)+head(fold2_sds,num_models), code=3, angle=90, len=0.02, col=colors_base_2, lwd=1)
    arrows(x0=x+3*offset+0.01, x1=x+3*offset+0.01, y0=tail(fold2_avgs,num_models)-tail(fold2_sds,num_models), y1=tail(fold2_avgs,num_models)+tail(fold2_sds,num_models), code=3, angle=90, len=0.02, col= colors_paper_2, lwd=1)

    arrows(x0=x+4*offset-0.01, x1=x+4*offset-0.01, y0=head(fold3_avgs,num_models)-head(fold3_sds,num_models), y1=head(fold3_avgs,num_models)+head(fold3_sds,num_models), code=3, angle=90, len=0.02, col=colors_base_3, lwd=1)
    arrows(x0=x+4*offset+0.01, x1=x+4*offset+0.01, y0=tail(fold3_avgs,num_models)-tail(fold3_sds,num_models), y1=tail(fold3_avgs,num_models)+tail(fold3_sds,num_models), code=3, angle=90, len=0.02, col= colors_paper_3, lwd=1)

    # draw total mean of every model
    points(x-0.01, head(total_avgs, num_models), col= colors_base, pch=16, cex=0.6)
    text(x-0.23, head(total_avgs, num_models), col= colors_base, label=round(head(total_avgs, num_models), digits=3), cex=0.6)
    points(x+0.01, tail(total_avgs, num_models), col= colors_paper, pch=17, cex=0.6)
    text(x-0.21, tail(total_avgs, num_models), col= colors_paper, label=round(tail(total_avgs, num_models), digits=3),cex=0.6)
    # draw means of every fold and every model
    points(x+offset-0.01, head(fold0_avgs,num_models), col=colors_base_0, pch=16,cex=0.6)
    points(x+offset+0.01, tail(fold0_avgs,num_models), col= colors_paper_0, pch=17,cex=0.6)

    points(x+2*offset-0.01, head(fold1_avgs,num_models), col=colors_base_1, pch=16,cex=0.6)
    points(x+2*offset+0.01, tail(fold1_avgs,num_models), col= colors_paper_1, pch=17,cex=0.6)

    points(x+3*offset-0.01, head(fold2_avgs,num_models), col=colors_base_2, pch=16,cex=0.6)
    points(x+3*offset+0.01, tail(fold2_avgs,num_models), col= colors_paper_2, pch=17,cex=0.6)

    points(x+4*offset-0.01, head(fold3_avgs,num_models), col=colors_base_3, pch=16,cex=0.6)
    points(x+4*offset+0.01, tail(fold3_avgs,num_models), col= colors_paper_3, pch=17,cex=0.6)

    # draw means of CNN
    for(i in 1:5)
    {
        points(tail(x2,1)+(i-1)*offset, cnn_avgs[i], pch=16, cex=0.6)
    }
    text(tail(x2,1)-0.2, cnn_avgs[1], label=round(cnn_avgs[1], digits=3), cex=0.6)
    par(cex=0.5)
    # label axis
    splits = list("total", "fold 0", "fold 1", "fold 2", "fold 3")
    label_location=c()
    for(i in x2)
    {
        label_location = c(label_location,seq(i, i+0.4, 0.1))
    }
    axis(1, at=label_location, labels=rep(splits, num_models+1) , las=3,lwd.ticks=0.5)
    par(cex=0.5)
    axis(3, at=(1:(num_models+1))+0.2, labels=c(models, "CNN"), cex=0.1, lwd.ticks=0.5)
    par(cex=1)
    # dev.copy(pdf, "Results_all_GNNs.pdf")
    # dev.off()
}

# setwd("/home/admin1/Desktop/MasterProject/GNNpT1/GNNpT1")

plot_Results()





# Rscript plotResults.r