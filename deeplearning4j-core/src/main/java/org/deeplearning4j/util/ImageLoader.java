/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.util;

import org.deeplearning4j.plot.FilterRenderer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

/**
 * Image loader for taking images and converting them to matrices
 * @author Adam Gibson
 *
 */
public class ImageLoader {

    private int width = -1;
    private int height = -1;


    public ImageLoader() {
        super();
    }

    public ImageLoader(int width, int height) {
        super();
        this.width = width;
        this.height = height;
    }

    public INDArray asRowVector(File f) throws Exception {
        return ArrayUtil.toNDArray(flattenedImageFromFile(f));
    }


    /**
     * Slices up an image in to a mini batch.
     *
     * @param f the file to load from
     * @param numMiniBatches the number of images in a mini batch
     * @param numRowsPerSlice the number of rows for each image
     * @return a tensor representing one image as a mini batch
     */
    public INDArray asImageMiniBatches(File f,int numMiniBatches,int numRowsPerSlice) {
        try {
            INDArray d = asMatrix(f);
            INDArray f2 = Nd4j.create(numMiniBatches, numRowsPerSlice, d.columns());
            return f2;
        }catch(Exception e) {
            throw new RuntimeException(e);
        }

    }

    public INDArray asMatrix(File f) throws IOException {
        return ArrayUtil.toNDArray(fromFile(f));
    }

    public int[] flattenedImageFromFile(File f) throws Exception {
        return ArrayUtil.flatten(fromFile(f));
    }

    public int[][] fromFile(File file) throws IOException {
        BufferedImage image = ImageIO.read(file);
        if (height > 0 && width > 0)
            image = toBufferedImage(image.getScaledInstance(height, width, Image.SCALE_SMOOTH));
        Raster raster = image.getData();
        int w = raster.getWidth(), h = raster.getHeight();
        int[][] ret = new int[w][h];
        for (int i = 0; i < w; i++)
            for (int j = 0; j < h; j++)
                ret[i][j] = raster.getSample(i, j, 0);

        return ret;
    }


    /**
     * Convert the given image to an rgb image
     * @param arr the array to use
     */
    public static  BufferedImage toBufferedImageRGB(INDArray arr) {
        if(arr.rank() < 3)
            throw new IllegalArgumentException("Arr must be 3d");
        BufferedImage image = new BufferedImage(arr.size(-2), arr.size(-1), BufferedImage.TYPE_INT_ARGB);

        FilterRenderer renderer = new FilterRenderer();
        try {
            for(int i = 0; i < arr.slices(); i++)
                renderer.renderFilters(arr.slice(i),"/home/agibsonccc/Desktop/renderold" + i + ".png",28,28,28);
        } catch (Exception e) {
            e.printStackTrace();
        }
        for (int i = 0; i < image.getWidth(); i++) {
            for (int j = 0; j < image.getHeight(); j++) {
                //  double patch_normal = (  column.getDouble(0) - col_min ) / ( col_max - col_min + 0.000001f );

                int r = 255 * Math.abs(arr.slice(0).getInt(i, j));
                int g = 255 * Math.abs(arr.slice(1).getInt(i, j));
                int b = 255 * Math.abs(arr.slice(2).getInt(i, j));
                int col = (r << 16) | (g << 8) | b;
                image.setRGB(i,j,col);
            }
        }

        return image;

    }

    public static BufferedImage toImage(INDArray matrix) {
        BufferedImage img = new BufferedImage(matrix.size(-2), matrix.size(-1), BufferedImage.TYPE_INT_ARGB);
        if(matrix.isMatrix()) {
            WritableRaster r = img.getRaster();
            int[] equiv = new int[matrix.length()];
            for(int i = 0; i < equiv.length; i++) {
                equiv[i] = (int) matrix.getScalar(i).getDouble(i);
            }


            r.setDataElements(0,0,matrix.rows(),matrix.columns(),equiv);
        }

        else {

        }
        return img;
    }


    /**
     * Converts a given Image into a BufferedImage
     *
     * @param img The Image to be converted
     * @return The converted BufferedImage
     */
    public static BufferedImage toBufferedImage(Image img)
    {
        if (img instanceof BufferedImage)
        {
            return (BufferedImage) img;
        }

        // Create a buffered image with transparency
        BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_ARGB);

        // Draw the image on to the buffered image
        Graphics2D bGr = bimage.createGraphics();
        bGr.drawImage(img, 0, 0, null);
        bGr.dispose();

        // Return the buffered image
        return bimage;
    }

}