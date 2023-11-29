package com.github.jbaiter.kenlm;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Resources {
    public static Path forResource(URL resource) {
        if (resource == null) {
            throw new NullPointerException("unknown path");
        }

        if (!"file".equals(resource.getProtocol())) {
            throw new IllegalArgumentException("resource " + resource + " is not a file");
        }

        try {
            return Paths.get(resource.toURI());
        } catch (URISyntaxException e) {
            throw new IllegalArgumentException(e);
        }
    }

    public static Path resource(String name) {
        return forResource(Resources.class.getResource(name));
    }
}
