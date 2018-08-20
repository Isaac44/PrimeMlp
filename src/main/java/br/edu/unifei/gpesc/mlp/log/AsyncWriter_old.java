/*
 * Copyright (C) 2015 - GEPESC - Universidade Federal de Itajuba
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package br.edu.unifei.gpesc.mlp.log;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.Charset;
import java.util.LinkedList;
import java.util.concurrent.ExecutorService;

/**
 *
 * @author Isaac Caldas Ferreira
 */
public class AsyncWriter_old {

    private static final int LIMIT = 8192;

    protected static final Charset UTF8 = Charset.forName("UTF-8");

    private final OutputStream mOutputStream;

    private LinkedList<ByteArray> mWriteList = new LinkedList<ByteArray>();
    private int mCurrentLength;

    private final ExecutorService mExecutor;

    public AsyncWriter_old(ExecutorService executor, File file) throws FileNotFoundException {
        mOutputStream = new BufferedOutputStream(new FileOutputStream(file), LIMIT);
        mExecutor = executor;
    }

    public void append(byte[] bytes) {
        mWriteList.add(new ByteArray(bytes));

        // added length
        mCurrentLength += bytes.length;

        if (mCurrentLength >= LIMIT) {
            // async write
            AsyncWrite log = new AsyncWrite(mWriteList);

            if (mExecutor != null) {
                mExecutor.execute(log);
            } else {
                log.run();
            }

            // reset
            mCurrentLength = 0;
            mWriteList = new LinkedList<ByteArray>();
        }
    }

    public void append(String str) {
        append(str.getBytes(UTF8));
    }

    public void close() {
        new AsyncWrite(mWriteList).run();
        try {
            mOutputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public class AsyncWrite implements Runnable {
        private final LinkedList<ByteArray> mWriteList;

        public AsyncWrite(LinkedList<ByteArray> writeList) {
            mWriteList = writeList;
        }

        @Override
        public void run() {

            for (ByteArray array : mWriteList) {
                try {
                    mOutputStream.write(array.bytes);
                }
                catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static class ByteArray {

        public final byte[] bytes;

        public ByteArray(byte[] bytes) {
            this.bytes = bytes;
        }
    }

}
