SELECT * FROM company_name AS cn, movie_companies AS mc, movie_keyword AS mk WHERE mc.company_id = cn.id AND cn.id = mc.company_id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id;