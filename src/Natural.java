class Natural extends Integer {
    Natural(int value) {
        super(value);
        if (value <= 0)
            throw new IllegalArgumentException("Natural numbers must be positive.");
    }
}