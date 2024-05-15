public class Main {
    public static void main(String[] args) {
        IntegerField a = new IntegerField(6);
        IntegerField b = new IntegerField(3);

        IntegerField sum = a.add(b);
        IntegerField product = a.multiply(b);
        IntegerField difference = a.subtract(b);
        IntegerField quotient = a.divide(b);

        System.out.println("a + b = " + sum);           // Outputs: a + b = 9
        System.out.println("a * b = " + product);       // Outputs: a * b = 18
        System.out.println("a - b = " + difference);    // Outputs: a - b = 3
        System.out.println("a / b = " + quotient);      // Outputs: a / b = 2

        RationalField r1 = new RationalField(1, 3);
        RationalField r2 = new RationalField(1, 2);
        RationalField sum1 = r1.add(r2); // Should simplify to 1
        RationalField product1 = r1.multiply(r2); // Should be 1/4
        RationalField difference1 = r1.subtract(r2); // Should be 0/1 or 0
        RationalField quotient1 = r1.divide(r2); // Should be 1

        System.out.println("r1 + r2 = " + sum1); // Outputs: r1 + r2 = 1
        System.out.println("r1 * r2 = " + product1); // Outputs: r1 * r2 = 1/4
        System.out.println("r1 - r2 = " + difference1); // Outputs: r1 - r2 = 0
        System.out.println("r1 / r2 = " + quotient1); // Outputs: r1 / r2 = 1
    }
}