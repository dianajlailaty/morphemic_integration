package org.activeeon.morphemic.service;

import org.apache.commons.io.IOUtils;
import org.apache.log4j.Logger;

import java.io.*;

public class TemporaryFilesHelper {

    private static final String TMP_SYS_PROPERTY = "java.io.tmpdir";

    private static final String TMP_DIRECTORY;

    private static final File TMP_DIRECTORY_FILE;

    private static final Logger LOGGER = Logger.getLogger(TemporaryFilesHelper.class);

    static {
        TMP_DIRECTORY = ((System.getProperty(TMP_SYS_PROPERTY).endsWith(File.separator)) ?
                System.getProperty(TMP_SYS_PROPERTY) + "proactive_tmp" :
                System.getProperty(TMP_SYS_PROPERTY) + File.separator + "proactive_tmp");
        TMP_DIRECTORY_FILE = new File(TMP_DIRECTORY);
        boolean result = TMP_DIRECTORY_FILE.mkdirs();
        if (result) {
            LOGGER.info("Temporary directory created successfully");
        } else {
            LOGGER.warn("Temporary directory couldn't be created");
        }
    }

    public static File createTempFile(String prefix, String suffix) throws IOException {
        File newFile = File.createTempFile(prefix, suffix, TMP_DIRECTORY_FILE);
        newFile.deleteOnExit();
        LOGGER.info("Temporary file " + newFile.getAbsolutePath() + " created successfully");
        return newFile;
    }

    public static File createTempFile(String prefix, String suffix, InputStream inStream) throws IOException {
        File newFile = createTempFile(prefix, suffix);
        byte[] buffer = new byte[inStream.available()];
        inStream.read(buffer);
        try (OutputStream outStream = new FileOutputStream(newFile)) {
            outStream.write(buffer);
            IOUtils.closeQuietly(inStream);
            IOUtils.closeQuietly(outStream);
        }
        LOGGER.info("Temporary file " + newFile.getAbsolutePath() + " filled successfully with inputStream");
        return newFile;
    }

    public static File createTempFile(String filename, InputStream inStream) throws IOException {
        String prefix = filename.split("[.]")[0];
        String suffix = filename.substring(filename.indexOf("."));
        return createTempFile(prefix, suffix, inStream);
    }

    public static File createTempFileFromResource(String filename) throws IOException {
        return createTempFile(filename.trim().substring(filename.lastIndexOf(File.separator)+1),
                TemporaryFilesHelper.class.getResourceAsStream(filename));
    }

    public static void delete(File fileToDelete) {
        if (fileToDelete != null) {
            boolean result = fileToDelete.delete();
            if (result) {
                LOGGER.info("Temporary file deleted successfully");
            } else {
                LOGGER.warn("Temporary file couldn't be deleted");
            }
        }
    }
}
