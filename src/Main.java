public class Main {
    public static void main(String[] args) {
        Integer integer = new Integer(2);
        Rational rational = new Rational(2,3);
        Ring sum = rational.add(integer);
        System.out.println(integer.power(integer));
        System.out.println(rational.power(integer));
    }
}
